import util

from gensim.models import word2vec
import gensim
import jieba
import re
import numpy as np

def split_word(content):
    words = jieba.lcut(content)
    words = remove_stop_word(words)
    return words

# 加载停用词表
stopwords = [line.strip() for line in open('../data/stop-words.txt', 'r', encoding='utf-8').readlines()]
def remove_stop_word(content_cut):
    outstr = []
    for word in content_cut:
        if word not in stopwords and word != '' and word != ' ':
            outstr.append(word)
    return outstr

r1 = '[0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~x]+'
def extraSourceFile():
    labels = []
    words = []
    labelMap = {}
    count = 0
    with open(util.sourceFile, encoding='utf8') as source:
        for line in source:
            s = line.split('_!_')
            label = s[1]
            title = re.sub(r1, ' ', s[3])
            title = split_word(title)
            if label not in labelMap.keys():
                labelMap[label] = len(labelMap)
            temp = np.zeros([15], dtype=np.int)
            temp[labelMap[label]] = 1
            labels.append(temp)
            words.append(title)

            count += 1
            if count % 1000 == 0:
                print(count)

    return labels, words, labelMap

def model_train(sentences, save_model_name='word2vec'):
    # 训练skip-gram模型;
    model = gensim.models.Word2Vec(sentences, size=128, window=5, iter=10, min_count=2)
    # 保存模型，以便重用
    model.wv.save(util.modelPath + save_model_name + '.model')

if __name__ == '__main__':
    labels, words, labelMap = extraSourceFile()

    if util.debug:
        print(labelMap)
        for i in range(100):
            print(labels[i], words[i])

    model_train(words)