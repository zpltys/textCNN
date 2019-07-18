import util

import gensim
import jieba
import re
import numpy as np
import h5py
import zhconv
import pickle

def split_word(content):
    content = zhconv.convert(content, 'zh-cn')
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
    word2index = {'': 0}
    count = 0

    oriwords = []
    with open(util.sourceFile, encoding='utf8') as source:
        for line in source:
            s = line.split('_!_')
            label = s[1]
            if label not in labelMap.keys():
                labelMap[label] = len(labelMap)
            temp = np.zeros([15], dtype=np.int)
            temp[labelMap[label]] = 1
            labels.append(temp)

            title = re.sub(r1, ' ', s[3])
            title = split_word(title)
            oriwords.append(title)

            sentence = sentence_transform(word2index, title)
            words.append(sentence)

            count += 1
            if count % 10000 == 0:
                print(count)

    return labels, words, labelMap, word2index, oriwords

def model_train(sentences, save_model_name='word2vec'):
    # 训练skip-gram模型;
    model = gensim.models.Word2Vec(sentences, size=128, window=5, iter=10, min_count=2)
    # 保存模型，以便重用
    model.wv.save(util.modelPath + save_model_name + '.model')

def sentence_transform(word2index, sentence):
    indexs = []
    for word in sentence:
        if word not in word2index.keys():
            word2index[word] = len(word2index)
        indexs.append(word2index[word])

    if len(indexs) > util.max_len:
        indexs = indexs[:util.max_len]
    while len(indexs) < util.max_len:
        indexs.append(0)
    return indexs


def sampleSplit(labels, words):
    length = len(labels)
    rand = np.random.random(length)
    trainX = np.array([words[i] for i in range(length) if rand[i] <= 0.7])
    trainY = np.array([labels[i] for i in range(length) if rand[i] <= 0.7])

    testX = np.array([words[i] for i in range(length) if 0.7 < rand[i] <= 0.85])
    testY = np.array([labels[i] for i in range(length) if 0.7 < rand[i] <= 0.85])

    validX = np.array([words[i] for i in range(length) if 0.85 < rand[i]])
    validY = np.array([labels[i] for i in range(length) if 0.85 < rand[i]])

    return trainX, trainY, testX, testY, validX, validY


def dumpMessage(trainX, trainY, testX, testY, validX, validY, word2index):
    h5File = h5py.File(util.dataPath + 'TrainTest.h5py', 'w')

    h5File.create_dataset('train_X', data=trainX)
    h5File.create_dataset('train_Y', data=trainY)

    h5File.create_dataset('valid_X', data=validX)
    h5File.create_dataset('valid_Y', data=validY)

    h5File.create_dataset('test_X', data=testX)
    h5File.create_dataset('test_Y', data=testY)
    h5File.close()

    pickleFile = open(util.dataPath + 'word2index.pickle', 'wb+')
    pickleFile.write(pickle.dumps(word2index))
    pickleFile.close()

if __name__ == '__main__':
    labels, words, labelMap, word2index, oriwords = extraSourceFile()
    #print(oriwords[0:10])
    #model_train(oriwords)

    trainX, trainY, testX, testY, validX, validY = sampleSplit(labels, words)

    print('trainX size:', len(trainX))
    print('trainY size:', len(trainY))

    print('testX size:', len(testX))
    print('testY size:', len(testY))

    print('validX size:', len(validX))
    print('validY size:', len(validY))

    dumpMessage(trainX, trainY, testX, testY, validX, validY, word2index)
