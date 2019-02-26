# -*- coding:utf-8 -*-
# author: adowu
import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig
from TextCNN import model_fn
from input import Input


def run():
    params = dict()
    load_config(params)
    enrich_params(params)
    log_level = 'tf.logging.{}'.format(str(params.get('log_level')).upper())
    tf.logging.set_verbosity(eval(log_level))
    config = load_sess_config(params)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params.get('model_dir'),
        config=config,
        params=params
    )

    input = Input(params)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input.input_fn(mode=tf.estimator.ModeKeys.TRAIN, vocabs=params.get('vocabs')),
        max_steps=params.get('max_steps', None),
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input.input_fn(mode=tf.estimator.ModeKeys.EVAL, vocabs=params.get('vocabs')),
        throttle_secs=params.get('throttle_secs'),
    )

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    predictions = estimator.predict(
        input_fn=lambda: input.input_fn(mode=tf.estimator.ModeKeys.PREDICT, vocabs=params.get('vocabs')),
    )

    for index, prediction in enumerate(predictions):
        print('label1 {} : label2 {}'.format(prediction[0], prediction[1]))


def load_config(params: dict):
    params['vocab_size'] = 10000
    params['embedding_size'] = 128
    params['learning_rate'] = 1e-3
    params['kernels'] = [3, 4, 5]
    params['filters'] = 128
    params['sequence_max_length'] = 30
    params['dropout'] = 0.5
    params['num_class'] = 2

    params['data_dir'] = './data'
    params['model_dir'] = 'file'
    params['vocab_file'] = './data/vocab.txt'
    params['feature_name'] = 'input'
    params['batch_size'] = 32
    params['max_steps'] = 1000
    params['log_level'] = 'info'

    params['gpu_cores'] = '1'
    params['allow_growth'] = True
    params['per_process_gpu_memory_fraction'] = 0.9
    params['allow_soft_placement'] = True

    params['save_checkpoints_steps'] = 100
    params['keep_checkpoint_max'] = 3
    params['log_step_count_steps'] = 100
    params['throttle_secs'] = 10


def load_sess_config(params):
    """
    :param params: some gpu & session config settings
    :return: run configurations
    """
    if params.get('gpu_cores'):
        #   gpu mode
        tf.logging.warn('using device: {}'.format(params.get('gpu_cores')))
        gpu_options = tf.GPUOptions(
            allow_growth=params.get('allow_growth'),
            visible_device_list=params.get('gpu_cores'),
            per_process_gpu_memory_fraction=params.get('per_process_gpu_memory_fraction')
        )

    else:
        gpu_options = tf.GPUOptions(
            allow_growth=params.get('allow_growth'),
        )
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=params.get('allow_soft_placement'),

    )
    config = RunConfig(
        save_checkpoints_steps=params.get('save_checkpoints_steps'),
        keep_checkpoint_max=params.get('keep_checkpoint_max'),
        log_step_count_steps=params.get('log_step_count_steps'),
        session_config=session_config

    )
    return config


def enrich_params(params):
    vocab_path = params.get('vocab_file')
    vocabs = list()
    vocabs.append('PAD')
    vocabs.append('UNK')
    data = open(vocab_path, 'r', encoding='utf-8')
    for line in data:
        l = line.split('\t')
        if len(l) == 2:
            vocabs.append(l[0])
            if len(vocabs) == params.get('vocab_size', 50000):
                tf.logging.info('vocab size is {}'.format(params.get('vocab_size', 50000)))
                break
    params['vocabs'] = vocabs

if __name__ == '__main__':
    run()
