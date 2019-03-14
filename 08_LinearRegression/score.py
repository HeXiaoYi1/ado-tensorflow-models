# -*- coding:utf-8 -*-
# author: adowu

from numpy import genfromtxt
from sklearn import linear_model

# from main.normalCalculator import normal_calculate, InputData

dataPath = "data/reduce_sqa_3.txt"
deliveryData = genfromtxt(dataPath,delimiter=',')
print(deliveryData)
# ------------------------准备数据---------------------------
x= deliveryData[:,:-1]
y = deliveryData[:,-1]
# --------------------------训练模型-------------------------
lr = linear_model.LinearRegression()
lr.fit(x, y)

coef_ = lr.coef_
print(coef_)
print(lr.intercept_)

pos_i = 0
x_ = [i for i in lr.predict(x)]
len__ = x_.__len__()
for i in range(0, len__):
    if abs(y[i] - x_[i]) < 0.2:
        pos_i += 1

print(pos_i * 1.0 / len__)
print(pos_i)


# title = '什么人要参加统考？'
# stan_ans = '教鱼部批准的试点高校和中央电大“人才培养模式改革和开放教育试点”项目中自2004年3月1日（含3月1日）以后入学的本科层次网络学历教育的学生（含高中起点本科、专科起点本科）应参加试点高校网络教育统考。对高中起点专科和研究生教育层次的学生暂不进行统考。'
# stu_ans = '教育部批准的试点高校和中央电大“人才培养模式改革和开放教育试点项目中，自2004年3月1日（含3月1日）以后入学的本科层次网络学历教育的学生（含高中起点本科，专科起点本科）应参加试点高校网络教育统考。对高中起点专科和研究生教育层次的学生暂不进行统考。'
#
#
# try:
#     calculate = normal_calculate(inputData=InputData(title=title, stand_ans=stan_ans, stu_ans=stu_ans))
# except Exception as e:
#     print(title, stu_ans)
#
#
# x = [calculate.f_base_sim, -(1 - calculate.smoothness_rate), calculate.length_gap, -calculate.error_rate,
#      calculate.bleu]
#
# predict = lr.predict([x])
# print(predict)





