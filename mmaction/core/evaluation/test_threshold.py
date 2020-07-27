# -*- coding: utf-8 -*-
# coding: utf-8

import numpy as np
import torch
#import torchsnooper
import json
import csv, codecs
import sys
import os

"""
Mapping and calculate Percision/Recall with saved scores and labels.
Save Percision/Recall in csv.
"""

threshold_top1 = 'top1'
if sys.argv[1] == 'top1':
    threshold_top1 = 'top1'
elif sys.argv[1] == 'threshold':
    threshold_top1 = 'threshold'


# 现在直接读key value都是汉字的json会报错，key是汉字value是数字就没有问题
dict1_101to0_45 = {
  "猫": ["猫","宠物","动物"],
  "小孩": ["小孩","人物"],
  "街道": ["建筑","城市风景"],
  #"旅行出游": ["旅行出游"],
  "冲浪": ["人物","运动"],
  "KTV": ["活动聚会","人物"],
  "钢琴": ["钢琴"],
  "茶": ["草","植物"],
  "狗": ["狗","宠物","动物"],
  "兔子": ["动物"],
  "仓鼠": ["宠物","动物"],
  "美女": ["人物"],
  "刺身": ["人物"],
  "拳击": ["人物"],
  "舞蹈": ["人物","舞蹈","运动"],
  "树": ["树","植物"],
  "鹦鹉": ["宠物","鸟"],
  "水域": ["自然风景"],
  "面包": ["美食"],
  "岛屿": ["自然风景"],
  "游戏": ["活动聚会","游戏"],
  "山谷": ["山","自然风景"],
  "吉他": ["吉他"],
  "沙漠": ["自然风景"],
  "婚礼": ["活动聚会","人物","婚礼"],
  "沙滩": ["自然风景"],
  "健身房": ["健身房"],
  "天空": ["自然风景","天空"],
  "跳水": ["人物","运动"],
  "滑雪": ["人物","运动"],
  "汤圆": ["美食"],
  "宠物": ["宠物","动物"],
  "帅哥": ["人物"],
  "火锅": ["美食"],
  "包子": ["美食"],
  "建筑": ["建筑","城市风景"],
  "山峰": ["山","自然风景"],
  "羽毛球": ["人物","运动"],
  "牛奶": ["美食"],
  "草": ["草","植物"],
  "煎蛋": ["美食"],
  "寿司": ["美食"],
  "足球": ["人物","运动"],
  "聚餐": ["聚餐","活动聚会","人物"],
  "蛋挞": ["美食"],
  "饺子": ["美食"],
  "滑冰": ["人物","运动"],
  "草原": ["草","植物"],
  "薯片": ["美食"],
  "老人": ["人物"],
  "桌游": ["活动聚会","人物"],
  "花": ["花","植物"],
  "瀑布": ["自然风景"],
  "篮球": ["人物","运动"],
  "炒面": ["美食"],
  "跆拳道": ["人物","运动"],
  "三明治": ["美食"],
  "馒头": ["美食"],
  "沙拉": ["美食"],
  "冰淇淋": ["美食"],
  "咖啡": ["美食"],
  "烟花": ["烟花"],
  "炒饭": ["美食"],
  "三角粽子": ["美食"],
  "红酒": ["美食"],
  "生日": ["生日"],
  "雪景": ["雪景","自然风景"],
  "潜水": ["人物","运动"],
  "日出日落": ["日出日落","自然风景"],
  "乒乓球": ["人物","运动"],
  "演唱会": ["人物","演唱会"],
  "动感单车": ["人物","健身房"],
  "香肠": ["美食"],
  "面条": ["美食"],
  "森林": ["树","自然风景"],
  "披萨": ["美食"],
  "高尔夫": ["人物","运动"],
  "奶油蛋糕": ["美食","甜点"],
  "月饼": ["美食"],
  "旅行": ["旅行出游"],
  "跳高": ["人物","运动"],
  "棒球": ["人物","运动"],
  "汤": ["美食"],
  "跑步机": ["人物","运动"],
  "米饭": ["美食"],
  "甜甜圈": ["美食","甜点"],
  "台球": ["人物","运动"],
  "击剑": ["人物","运动"],
  "水球": ["人物","运动"],
  "车流": ["车"],
  "跳远": ["人物","运动"],
  "跨栏": ["人物","运动"],
  "健身球": ["人物","运动","健身房"],
  "粥": ["美食"],
  "农场": ["动物"],
  "跑步": ["人物","运动"],
  "攀岩": ["人物","运动"],
  "游泳": ["人物","运动"],
  "水果奶油蛋糕": ["美食","甜点"],
  "骑单车": ["人物","运动"],
  "排球": ["人物","运动"]
}

# 目前结果和掉包算出来的结果一样
# load npy as scores and y_true
# scores = np.load("prob_scores.npy")
def logistic(z):
    return 1 / (1 + np.exp(-z))
scores = np.load("prob_scores_raw.npy")
scores = logistic(np.array(scores/5))
y_true = np.load("prob_y_true.npy")
# transform scores to y_pred
if threshold_top1 == 'threshold':
    threshold = 0.92 * np.ones(400) 
    #threshold[1] = 0.91  
    y_pred = np.where(scores > threshold,1,0)  
elif threshold_top1 == 'top1':
    pred = np.argmax(scores, axis=1)
    y_pred_max = np.zeros_like(scores)
    for i in range(len(pred)):
        y_pred_max[i,pred[i]] = 1
    y_pred = y_pred_max


# mapping y_pred to target y_pred
def read_json(jsonfile):
    with open(jsonfile,'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict
dict0_45 = read_json('/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/0_45.json')
dict1_101 = read_json('/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/1_101.json')
# dict1_101to0_45 = read_json('/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/1_101to0_45.json')
y_pred_46 = np.zeros_like(y_pred)
for _ in dict1_101to0_45.keys():
    i = dict1_101[_]
    for __ in dict1_101to0_45[_]:
        j = dict0_45[__]
        index = np.where(y_pred[:,i] == 1)
        y_pred_46[index,j] = 1
y_pred = y_pred_46


# Calculate Percision/Recall with y_pred and y_true
TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)),axis=0)   
FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)),axis=0)   
FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)),axis=0)
P = TP/(TP+FP)
R = TP/(TP+FN)
print('Percisions',P)
print('Recalls',R)
for _ in dict0_45.keys():
    print("{:.10f}".format(P[dict0_45[_]]),'\t ',"{:.10f}".format(R[dict0_45[_]]),'\t ',str(_))


# test_threshold.csv
# test_top1.csv
# Save Percision/Recall in csv.
if threshold_top1 == 'threshold':
    os.remove('./test_threshold.csv')
    with open('./test_threshold.csv','a+',encoding="gbk") as csvfile:
        writeCSV = csv.writer(csvfile)
    for _ in dict0_45.keys():
        writeCSV.writerow([_,"{:.10f}".format(P[dict0_45[_]]),"{:.10f}".format(R[dict0_45[_]])])
elif threshold_top1 == 'top1':
    os.remove('./test_top1.csv')
    with open('./test_top1.csv','a+',encoding="gbk") as csvfile:
        writeCSV = csv.writer(csvfile)
    for _ in dict0_45.keys():
        writeCSV.writerow([_,"{:.10f}".format(P[dict0_45[_]]),"{:.10f}".format(R[dict0_45[_]])])


# print total_Percision/total_Recall.
TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1))) 
FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  
FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0))) 
P=TP/(TP+FP) 
R=TP/(TP+FN) 
print('total_Percisions',P)
print('total_Recalls',R)


# print classification_report(y_true, y_pred).
from sklearn.metrics import classification_report
print('classification_report(y_true, y_pred)',classification_report(y_true, y_pred))

