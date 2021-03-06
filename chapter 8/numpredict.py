import math

from random import random, randint

def wineprice(rating, age):
    peak_age = rating - 50

    # 根据等级来计算价格
    price = rating / 2
    if age > peak_age:
        # 经过“峰值年”，后继5年里其品质将会变差
        price = price * (5 - (age - peak_age))
    else:
        # 价格在接近“峰值年”时会增加到原值的5倍
        price = price * (5 * (age + 1) / peak_age)
    if price < 0:
        price = 0
    return price


def wineset1():
    rows = []
    for i in range(300):
        # 随机生成年代和等级
        rating = random() * 50 + 50
        age = random() * 50

        # 得到一个参考价格
        price = wineprice(rating, age)

        # 增加“噪声”
        price *= (random() * 0.4 + 0.8)

        # 加入数据集
        rows.append({'input': (rating, age),
            'result': price})
    return rows


def euclidean(v1, v2):
    d = 0.0
    for i in range(len(v1)):
        d +=(v[i] - v[i])**2
    return math.sqrt(d)


def getdistances(data, vec1):
    distancelist = []
    for i in range(len(data)):
        vec2 = data[i]['input']
        distancelist.append((euclidean(vec1, vec2), i))
    distancelist.sort()
    return distancelist

