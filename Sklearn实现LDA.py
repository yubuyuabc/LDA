# %%

import jieba
import importlib, sys

importlib.reload(sys)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 沙瑞金不分开

jieba.suggest_freq('沙瑞金', True)

# %%

# -*- coding: utf-8 -*-

import jieba

for i in range(16):
    with open('./doc%d.txt' % (i + 1), 'r', encoding="utf-8") as f1:
        document = f1.read()
        document_cut = jieba.cut(document)
        result = ' '.join(document_cut)
        print(result)
        f1.close()
        with open('./result%d.txt' % (i + 1), 'w', encoding="utf-8") as f2:
            f2.write(result)
            f2.close()

# %%

with open('./ChineseStopWords.txt', 'r', encoding="utf-8") as f:
    line = f.read()
    line = line.split('","')
    f.close()

file_object = open('./stopwords.txt', 'w', encoding="utf-8")
for i in range(len(line)):
    file_object.write(line[i] + '\n')
file_object.close()

with open('./stopwords.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    f.close()

stopwords = []
for l in lines:
    stopwords.append(l.strip())
print(stopwords)

# %%

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

with open('./result1.txt', 'r', encoding="utf-8") as f:
    res1 = f.read()
    f.close()
with open('./result2.txt', 'r', encoding="utf-8") as f:
    res2 = f.read()
    f.close()
with open('./result3.txt', 'r', encoding="utf-8") as f:
    res3 = f.read()
    f.close()
with open('./result4.txt', 'r', encoding="utf-8") as f:
    res4 = f.read()
    f.close()
with open('./result5.txt', 'r', encoding="utf-8") as f:
    res5 = f.read()
    f.close()
with open('./result6.txt', 'r', encoding="utf-8") as f:
    res6 = f.read()
    f.close()
with open('./result7.txt', 'r', encoding="utf-8") as f:
    res7 = f.read()
    f.close()
with open('./result8.txt', 'r', encoding="utf-8") as f:
    res8 = f.read()
    f.close()
with open('./result9.txt', 'r', encoding="utf-8") as f:
    res9 = f.read()
    f.close()
with open('./result10.txt', 'r', encoding="utf-8") as f:
    res10 = f.read()
    f.close()
with open('./result11.txt', 'r', encoding="utf-8") as f:
    res11 = f.read()
    f.close()
with open('./result12.txt', 'r', encoding="utf-8") as f:
    res12 = f.read()
    f.close()
with open('./result13.txt', 'r', encoding="utf-8") as f:
    res13 = f.read()
    f.close()
with open('./result14.txt', 'r', encoding="utf-8") as f:
    res14 = f.read()
    f.close()
with open('./result15.txt', 'r', encoding="utf-8") as f:
    res15 = f.read()
    f.close()
with open('./result16.txt', 'r', encoding="utf-8") as f:
    res16 = f.read()
    f.close()

vector = TfidfVectorizer(stop_words=stopwords)
tfidf = vector.fit_transform([res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
                              res11, res12, res13, res14, res15, res16])

print(tfidf)

# %%

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

with open('./result1.txt', 'r', encoding="utf-8") as f:
    res1 = f.read()
    f.close()
with open('./result2.txt', 'r', encoding="utf-8") as f:
    res2 = f.read()
    f.close()
with open('./result3.txt', 'r', encoding="utf-8") as f:
    res3 = f.read()
    f.close()
with open('./result4.txt', 'r', encoding="utf-8") as f:
    res4 = f.read()
    f.close()
with open('./result5.txt', 'r', encoding="utf-8") as f:
    res5 = f.read()
    f.close()
with open('./result6.txt', 'r', encoding="utf-8") as f:
    res6 = f.read()
    f.close()
with open('./result7.txt', 'r', encoding="utf-8") as f:
    res7 = f.read()
    f.close()
with open('./result8.txt', 'r', encoding="utf-8") as f:
    res8 = f.read()
    f.close()
with open('./result9.txt', 'r', encoding="utf-8") as f:
    res9 = f.read()
    f.close()
with open('./result10.txt', 'r', encoding="utf-8") as f:
    res10 = f.read()
    f.close()
with open('./result11.txt', 'r', encoding="utf-8") as f:
    res11 = f.read()
    f.close()
with open('./result12.txt', 'r', encoding="utf-8") as f:
    res12 = f.read()
    f.close()
with open('./result13.txt', 'r', encoding="utf-8") as f:
    res13 = f.read()
    f.close()
with open('./result14.txt', 'r', encoding="utf-8") as f:
    res14 = f.read()
    f.close()
with open('./result15.txt', 'r', encoding="utf-8") as f:
    res15 = f.read()
    f.close()
with open('./result16.txt', 'r', encoding="utf-8") as f:
    res16 = f.read()
    f.close()

corpus = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
          res11, res12, res13, res14, res15, res16]
labels = ['碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传',
          '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖',
          '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑', '白马啸西风']
corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'categoray': labels})
Normalize_corpus = np.vectorize(corpus)
corpus_norm = corpus

Tf = TfidfVectorizer(use_idf=True)
Tf.fit(corpus_norm)
vocs = Tf.get_feature_names()
corpus_array = Tf.transform(corpus_norm).toarray()
corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)
# corpus_norm_df = corpus_norm_d.reindex([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]) index=list('abcdefghijklmn')
print(corpus_norm_df.head())

vector = TfidfVectorizer(stop_words=stopwords)
tfidf = vector.fit_transform(
    [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13, res14, res15, res16])

print(tfidf)

# %%

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
wordlist = vector.get_feature_names()  # 获取词袋模型中的所有词
# print(wordlist)

# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()

# print(len(weightlist[0]))
# print(len(weightlist[0]))
# print(len(weightlist[0]))
# print(weightlist)

# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
print("-------第res1段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res1:
        print(wordlist[j], weightlist[0][j])
print("-------第res2段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res2:
        print(wordlist[j], weightlist[1][j])
print("-------第res3段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res3:
        print(wordlist[j], weightlist[2][j])
print("-------第res4段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res4:
        print(wordlist[j], weightlist[3][j])
print("-------第res5段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res5:
        print(wordlist[j], weightlist[4][j])
print("-------第res6段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res6:
        print(wordlist[j], weightlist[5][j])
print("-------第res7段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res7:
        print(wordlist[j], weightlist[6][j])
print("-------第res8段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res8:
        print(wordlist[j], weightlist[7][j])
print("-------第res9段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res9:
        print(wordlist[j], weightlist[8][j])
print("-------第res10段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res10:
        print(wordlist[j], weightlist[9][j])
print("-------第res11段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res11:
        print(wordlist[j], weightlist[10][j])
print("-------第res12段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res12:
        print(wordlist[j], weightlist[11][j])
print("-------第res13段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res13:
        print(wordlist[j], weightlist[12][j])
print("-------第res14段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res14:
        print(wordlist[j], weightlist[13][j])
print("-------第res15段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res15:
        print(wordlist[j], weightlist[14][j])
print("-------第res16段文本的词语tf-idf权重------")
for j in range(len(wordlist)):
    if wordlist[j] in res16:
        print(wordlist[j], weightlist[15][j])

# %%

# 输出即为所有文档中各个词的词频向量。有了这个词频向量，我们就可以来做LDA主题模型了，选择主题数K=16
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=16, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
docres0 = lda.fit_transform(tfidf)
print(docres0)

lda_corpus = np.array(lda.fit_transform(corpus_array))
lda_corpus_one = np.zeros([lda_corpus.shape[0]])
lda_corpus_one[lda_corpus[:, 0] < lda_corpus[:, 1]] = 1
corpus_norm_df['lda_labels'] = lda_corpus_one
print(corpus_norm_df.head())

# %%

# 通过fit_transform函数，我们就可以得到文档的主题模型分布在docres中。而主题词分布则在lda.components_中。
# print(docres)
# print(lda.components_)
# 第六步：打印每个单词的主题的权重值
tt_matrix = lda.components_
for tt_m in tt_matrix:
    tt0_dict = [(name, tt) for name, tt in zip(wordlist, tt_m)]
    tt0_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    # 打印权重值大于0.6的主题词
    # tt0_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 1.0]
    # tt0_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.9]
    # tt0_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
    tt0_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0]
    print(tt0_dict)

# %%

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib

joblib.dump(vector, "TMvectorMat.model")  # 注意unigram保存的是模型不是矩阵
tfidfModel = TfidfTransformer().fit_transform(tfidf)
joblib.dump(tfidfModel, "TMtfidfMat.model")  # 注意tfidf保存的是模型不是矩阵
joblib.dump(tt0_dict, "tt0_dict-Tfidf.model")
tfidfdocres = TfidfTransformer().fit_transform(docres0)
joblib.dump(tfidfdocres, "docres0-tfidf.model")  # 注意tfidf保存的是模型不是矩阵
corpus_norm_df.head()

# %%

# 接着我们要把词转化为词频向量，注意由于LDA是基于词频统计的，因此一般不用TF-IDF来做文档特征。

from sklearn.feature_extraction.text import CountVectorizer

corpus = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
          res11, res12, res13, res14, res15, res16]
cntVector = CountVectorizer(stop_words=stopwords)

cntTf = cntVector.fit_transform(corpus)
print(cntTf)

# %%

# 输出即为所有文档中各个词的词频向量。有了这个词频向量，我们就可以来做LDA主题模型了，选择主题数K=16
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=16, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)
print(docres)

# %%

# 通过fit_transform函数，我们就可以得到文档的主题模型分布在docres中。而主题词分布则在lda.components_中。
# print(docres)
# print(lda.components_)
# 第六步：打印每个单词的主题的权重值
tt_matrix = lda.components_
for tt_m in tt_matrix:
    tt_dict = [(name, tt) for name, tt in zip(wordlist, tt_m)]
    tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    # 打印权重值大于0.6的主题词
    tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 1.0]
    # tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.9]
    # tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
    # tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0]
    print(tt_dict)

# %%

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib

joblib.dump(cntVector, "TMcntVectorMat.model")  # 注意unigram保存的是模型不是矩阵
cntTfModel = TfidfTransformer().fit_transform(cntTf)
joblib.dump(cntTfModel, "TMcntTfMat.model")  # 注意tfidf保存的是模型不是矩阵

joblib.dump(tt_dict, "tt_dict-cntTf.model")
cntTfdocres = TfidfTransformer().fit_transform(docres)
joblib.dump(cntTfdocres, "docres-cntTf.model")  # 注意tfidf保存的是模型不是矩阵

# %%


