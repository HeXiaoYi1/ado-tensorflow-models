#-*- coding:utf-8 -*-
# author: adowu

import tensorflow as tf
from tensorflow.contrib.estimator import multi_head
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.training import training_util
from tensorflow.contrib.layers import xavier_initializer
from ultra.classification.common.batch_norm import batch_norm
from ultra.classification.cnn_cail.parameters import user_params
from ultra.common.word2vec import word_embedding_initializer
from tensorflow.python.ops.lookup_ops import index_table_from_file

'''
  * Created by linsu on 2018/5/12.
  * mailto: lsishere2002@hotmail.com
'''

def linear(input_, output_size, scope=None):

    with tf.variable_scope(scope or "SimpleLinear"):
        dense_net = tf.layers.Dense(output_size, kernel_initializer=xavier_initializer(), activation=tf.nn.relu, kernel_regularizer=tf.nn.l2_loss)
        return dense_net(input_)


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


def model_fn(features, labels, mode: tf.estimator.ModeKeys, config: RunConfig, params: user_params):
    print("--- model_fn in %s" % mode)

    num_ps_replicas = config.num_ps_replicas if config else 0
    partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)

    with tf.variable_scope("cnn_classifcation", partitioner=partitioner, initializer=xavier_initializer()):
        logits,losses= inference(features, mode, params)

        def train_op_fn(loss):
            # loss_list = list()
            # loss_list.extend(losses)
            # loss_list.append(loss)
            # loss = tf.add_n(loss_list)
            return tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss,
                                                                                       global_step=training_util.get_global_step())

        #join_head = multi_head([params.acc_head, params.inprison_head, params.item_head],[1.0,0.01,1.1])
        join_head = multi_head([params.acc_head,  params.item_head])

        spec = join_head.create_estimator_spec(
            features, mode, logits, labels=labels, train_op_fn=train_op_fn)
        return spec


def inference(features, mode: tf.estimator.ModeKeys, params: user_params):
    losses = list()
    if params.embedding_file_path:
        padding = tf.get_variable("padding",shape = (1,params.embedding_size),dtype=tf.float32
                                  ,initializer= tf.zeros_initializer(tf.float32),trainable= False)
        unk = tf.get_variable("unk",shape = (1,params.embedding_size),dtype=tf.float32
                                  ,initializer=tf.random_uniform_initializer(-1, 1),trainable= True)
        embedding = tf.get_variable("embedding", shape=(params.feature_voc_known_embbeding_len, params.embedding_size)
                                    , initializer=word_embedding_initializer(params.embedding_file, include_word=False),
                                    trainable=True)

        if params.feature_voc_file_len - params.feature_voc_known_embbeding_len - 2 > 0:
            unknown_embedding = tf.get_variable("unknown_embedding",
                                                shape=(params.feature_voc_file_len - params.feature_voc_known_embbeding_len - 2, params.embedding_size),
                                                initializer=tf.random_uniform_initializer(-1, 1), dtype=tf.float32)
            embedding = tf.concat([padding, unk, embedding, unknown_embedding], axis=0)
        else:
            embedding = tf.concat([padding, unk, embedding], axis=0)

    else:
        padding = tf.get_variable("padding",shape = (1,params.embedding_size),dtype=tf.float32
                                  ,initializer= tf.zeros_initializer(tf.float32),trainable= False)
        embedding = tf.get_variable("embedding", shape=(params.feature_voc_file_len, params.embedding_size)
                                    , dtype=tf.float32, initializer=tf.random_uniform_initializer(-1, 1))
        embedding = tf.concat([padding,embedding],axis=0)

    table = index_table_from_file(
        vocabulary_file=params.feature_voc_file,
        num_oov_buckets=0,
        vocab_size=params.feature_voc_file_len,
        default_value=1,
        key_dtype=tf.string,
        name='{}_lookup'.format(params.feature_name))
    lookup = table.lookup(features[params.feature_name])

    if mode != tf.estimator.ModeKeys.PREDICT:
        lookup_dense = tf.sparse_tensor_to_dense(lookup,default_value=0)
    else:
        lookup_dense = lookup

    def truncate():
        return tf.slice(lookup_dense,begin=[0,0],size = [-1,params.truncate_sentence_len])
    def pad():
        return tf.concat([lookup_dense,tf.zeros([tf.gather(tf.shape(lookup_dense),indices=0)
                                                   ,tf.subtract(tf.constant(params.truncate_sentence_len)
                                                                ,tf.gather(tf.shape(lookup_dense),indices=1))]
                                                ,dtype=tf.int64)]
                         ,axis=1)

    lookup_dense = tf.cond(tf.greater_equal(tf.shape(lookup_dense)[1], tf.constant(params.truncate_sentence_len))
                           ,truncate
                           ,pad
                           )
    embedded_words = tf.nn.embedding_lookup(embedding, lookup_dense)

    sentence_embeddings_expanded = tf.cast(tf.expand_dims(embedded_words, -1), dtype=tf.float32)  # add channels

    pooled_outputs = list()

    for filter_size in params.filter_sizes:
        # size_after_conv = params.sentence_len - filter_size + 1
        size_after_conv = params.truncate_sentence_len - filter_size + 1
        with tf.variable_scope("filter_size_%s" % filter_size):

            filters = tf.get_variable("filters", shape=[filter_size, params.embedding_size, 1, params.filter_number], dtype=tf.float32)
            losses.append(tf.nn.l2_loss(filters))
            conv = tf.nn.conv2d(sentence_embeddings_expanded, filters, strides=[1, 1, 1, 1], padding="VALID",
                                name="conv")
            if params.use_batch_norm:
                conv = batch_norm(conv,mode,training_util.get_global_step(),params.filter_number,True)

            b = tf.get_variable("nolinear_bias", [params.filter_number], initializer=tf.zeros_initializer(), dtype=tf.float32)

            nolinear_logits = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(nolinear_logits, ksize=[1, size_after_conv, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")
            pooled = tf.reshape(pooled, (-1, params.filter_number))
            pooled_outputs.append(pooled)

    batch_sentence_feature = tf.concat(pooled_outputs, 1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_sentence_feature = tf.nn.dropout(batch_sentence_feature, keep_prob=params.keep_prob)

    # 增加一个fc层 ########################
    # middle_dense_net_accu = tf.layers.Dense(units=params.acc_classes * 2,kernel_initializer= xavier_initializer()
    #                              ,kernel_regularizer=tf.nn.l2_loss,name="middle_acc")
    # losses.extend(middle_dense_net_accu.losses)
    # middle_accu = middle_dense_net_accu(batch_sentence_feature)
    #
    # middle_dense_net_law = tf.layers.Dense(units=params.item_classes * 2,kernel_initializer= xavier_initializer()
    #                              ,kernel_regularizer=tf.nn.l2_loss,name="middle_acc")
    # losses.extend(middle_dense_net_law.losses)
    # middle_law = middle_dense_net_law(batch_sentence_feature)
    ######################################


    # 增加一个highway层 ###################
    middle_accu = highway(batch_sentence_feature, batch_sentence_feature.get_shape().as_list()[1], num_layers = 2, scope="highway_accu")
    middle_law = highway(batch_sentence_feature, batch_sentence_feature.get_shape().as_list()[1], num_layers = 2, scope="highway_law")
    #####################################


    dense_net = tf.layers.Dense( units=params.acc_classes,kernel_initializer= xavier_initializer()
                                 ,kernel_regularizer=tf.nn.l2_loss,name="acc")
    losses.extend(dense_net.losses)
    # acc_logits = dense_net(batch_sentence_feature)
    acc_logits = dense_net(middle_accu)


    dense_net = tf.layers.Dense( units=params.item_classes,kernel_initializer= xavier_initializer()
                                 ,kernel_regularizer=tf.nn.l2_loss,name = "item")
    losses.extend(dense_net.losses)
    # item_logits = dense_net(batch_sentence_feature)
    item_logits = dense_net(middle_law)



    #dense_net = tf.layers.Dense( units=params.inPrison_classes, kernel_initializer= xavier_initializer()
    #                            ,kernel_regularizer=tf.nn.l2_loss,name = "inprison")
    #losses.extend(dense_net.losses)
    #inprisoin_logits = dense_net(tf.concat([batch_sentence_feature,acc_logits,item_logits],-1))
    return {params.label_name_acc:acc_logits,params.label_name_item:item_logits},losses
