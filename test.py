import gensim
import util

def test1():
    word2vec_model = gensim.models.KeyedVectors.load(util.modelPath + 'word2vec.model')
    word2vec_dict = {}
    i = 0

    print(list(word2vec_model.vocab.keys())[0])
    print(word2vec_model['京城'])
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector

    print(word2vec_model.vectors[0])
    print(word2vec_dict['京城'])

def add(wc, v):
    wc[v] = v * 2

if __name__ == '__main__':
    w = {}
    add(w, 1)
    add(w, 2)
    print(w)