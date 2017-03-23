# stage1. 
#    截至目前，是通过训练文档中出现的特征及其对应的类别频率，来对新的特征的分类进行预测
#    通过设置初始概率为0.5，增强预测的平衡性

# stage2. 
#    为了预测整个文档的分类，引入了独立性假设：每个单词在指定分类上的概率，和其他单词
#    在该分类上的概率无关。进而引入贝叶斯定理：P(A|B) = P(B|A) * P(A) / P(B)
#    即：P(cat | doc) = P(doc | cat) * P(cat) / P(doc)
#    因为对于每个分类来说 P(doc)的值都是一样的，所以在计算上不考虑P(doc)，而是通过比较各分类下
#    P(doc | cat) * P(cat)的值来决定该文档doc属于哪个分类

# stage3. 
#    由于在现实任务中，对于各类错误（第1、2类错误）的倾向是不一样的，所以引入了阈值(threshold)的概念


import re
import math


def sampletrain(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick brown fox jumps', 'good')


def getwords(doc):
    splitter = re.compile("\\W*")
    # 根据非字母字符进行单词拆分
    words = [s.lower() for s in splitter.split(doc)
            if len(s) > 2 and len(s) < 20]

    return dict([(w, 1) for w in words])


class classifier:
    def __init__(self, getfeatures, filename=None):
        # 统计特征/分类组合的数量
        self.fc = {}
        # 统计每个分类中的文档数量
        self.cc = {}
        self.getfeatures = getfeatures
        self.thresholds = {}

    def setthreshold(self, cat, t):
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    # 增加对特征/分类组合的计数值
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # 增加对某一分类的计数值
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # 某一特征出现于某一分类中的次数
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    # 属于某一分类的内容项数量
    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0.0

    # 所有内容项的数量
    def totalcount(self):
        return sum(self.cc.values())

    # 所有分类的列表
    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        # 针对该分类为每个特征增加计数值
        for f in features:
            self.incf(f, cat)

        # 增加针对该分类的计数值
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat)==0:
            return 0
        #  特征在分类中出现的总次数，除以分类中包含内容项的总数
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # 计算当前的概率值
        basicprob = prf(f, cat)

        # 统计特征在所有分类中出现的次数
        totals = sum([self.fcount(f, cat) for cat in self.categories()])

        # 计算加权平均
        bp = (weight*ap + totals*basicprob) / (totals+weight)

        return bp

    def classify(self, item, default=None):
        probs = {}
        # 寻找概率最大的分类
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat
        # 确保概率值超出阈值*次大概率值
        for cat in probs:
            if cat==best:
                continue
            if probs[cat] * self.getthreshold(best) > probs[best]:
                # 'good'概率乘以阈值大于'bad'概率 -- 当best = 'bad'
                # 或者'bad'概率乘以阈值大于'good'概率 -- 当best = 'good'
                return default

        return best


class naivebayes(classifier):
    def docprob(self, item, cat):
        features = self.getfeatures(item)
        # 将所有特征的概率相乘
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return catprob * docprob


class fisherclassifier(classifier):
    def cprob(self, f, cat):
        # 特征在该分类中出现的频率
        clf = self.fprob(f, cat)
        if clf==0:
            return 0

        # 特征在所有分类中出现的频率
        freqsum = sum([self.fprob(f, c) for c in self.categories()])
        # 概率等于特征在该分类中出现的频率除以总体频率
        p = clf / freqsum

        return p



cl = naivebayes(getwords)
sampletrain(cl)
