import tensorflow as tf
import numpy as np
from textCNN_model import TextCNN
import pickle
import h5py
import os
import gensim
import util
from sklearn import metrics


def tfFlagConfig():
    FLAGS = tf.flags.FLAGS

    tf.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
    tf.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
    tf.flags.DEFINE_float("decay_rate", 1, "Rate of decay for learning rate.")  # 0.65一次衰减多少
    tf.flags.DEFINE_integer("sentence_len", 30, "max sentence length")
    tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
    tf.flags.DEFINE_boolean("is_training_flag", True, "is training.true:tranining,false:testing/inference")
    tf.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
    tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
    tf.flags.DEFINE_integer("num_filters", 128, "number of filters")  # 256--->512
    tf.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
    return FLAGS

filter_sizes = [5, 6, 7]


#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    FLAGS = tfFlagConfig()
    word2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data(util.dataPath + 'TrainTest.h5py', util.dataPath + 'word2index.pickle')
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:", vocab_size)

    num_examples, FLAGS.sentence_len = trainX.shape
    print("num_examples of training:", num_examples, " ;sentence_len:", FLAGS.sentence_len)

    #create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, 15, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate, FLAGS.sentence_len, vocab_size, FLAGS.embed_size)

        #Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(util.modelPath + "checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(util.modelPath))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            index2word = {v: k for k, v in word2index.items()}
            assign_pretrained_word_embedding(sess, index2word, vocab_size, textCNN)

        curr_epoch = sess.run(textCNN.epoch_step)
        #3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration + 1
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                feed_dict = {textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.8,
                             textCNN.is_training_flag: FLAGS.is_training_flag, textCNN.input_y: trainY[start:end]}

                curr_loss, lr, _ = sess.run([textCNN.loss_val, textCNN.learning_rate, textCNN.train_op], feed_dict)
                loss, counter = loss+curr_loss, counter+1
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" % (epoch, counter, loss / float(counter), lr))

                ########################################################################################################
                if start % (3000*FLAGS.batch_size) == 0:
                    eval_loss, f1_score, f1_micro, f1_macro = do_eval(sess, textCNN, vaildX, vaildY, 15)
                    print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch, eval_loss, f1_score, f1_micro, f1_macro))
                    # save model to checkpoint
                    save_path = util.modelPath + "model.ckpt"
                    print("Going to save model..")
                    saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################

            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, f1_score, f1_micro, f1_macro = do_eval(sess, textCNN, testX, testY, 15)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch,eval_loss,f1_score,f1_micro,f1_macro))
                #save model to checkpoint
                save_path = util.modelPath + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, f1_score, f1_micro, f1_macro = do_eval(sess, textCNN, testX, testY, 15, False)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (test_loss, f1_score, f1_micro, f1_macro))

    writer = tf.summary.FileWriter("../log/textCNN.log", sess.graph)
    writer.close()


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, num_classes, valid=True):
    if valid:
        evalX = evalX[0:3000]
        evalY = evalY[0:3000]
    number_examples = len(evalX)
    eval_loss, eval_counter= 0.0, 0
    batch_size = FLAGS.batch_size
    predict = []

    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        if end + batch_size > number_examples:
            end += batch_size
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y: evalY[start:end], textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        current_eval_loss, logits = sess.run(
            [textCNN.loss_val, textCNN.logits], feed_dict)
        predict += list(logits)
        eval_loss += current_eval_loss
        eval_counter += 1

    y_predict = [findMaxindex(y) for y in predict]
    y_true = [findMaxindex(y) for y in evalY]

    precision_macro = metrics.precision_score(y_true, y_predict, average='macro')
    precision_micro = metrics.precision_score(y_true, y_predict, average='micro')
    precision = (precision_macro + precision_micro) / 2.0

    recall_macro = metrics.recall_score(y_true, y_predict, average='macro')
    recall_micro = metrics.recall_score(y_true, y_predict, average='micro')
    recall = (recall_macro + recall_micro) / 2.0

    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    f1 = (f1_micro + f1_macro) / 2.0

    print("precision:", precision, " precision_macro:", precision_macro, " precision_micro:", precision_micro)
    print("recall:", recall, " recall_macro:", recall_macro, " recall_micro:", recall_micro)

    return eval_loss/float(eval_counter), f1, f1_micro, f1_macro


def findMaxindex(array):
    ans = 0
    val = array[0]
    for i in range(1, len(array)):
        if val < array[i]:
            val = array[i]
            ans = i
    return ans


def assign_pretrained_word_embedding(sess, index2word, vocab_size, textCNN, word2vec_model_path=util.modelPath+'word2vec.model'):
    print("using pre-trained word emebedding word2vec_model_path:", word2vec_model_path)
    wv = gensim.models.KeyedVectors.load(word2vec_model_path)


    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word. notice that the first two words are pad and unknown token
        word = index2word[i]  # get a word
        embedding = None
        try:
            embedding = wv[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None

        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.

    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


def load_data(cacheH5py, cachePickle):
    h5Data = h5py.File(cacheH5py, 'r')
    print("f_data.keys:", list(h5Data.keys()))
    train_X = np.array(h5Data['train_X'])
    print("train_X.shape:", train_X.shape)
    train_Y = np.array(h5Data['train_Y'])
    print("train_Y.shape:", train_Y.shape)
    vaild_X = np.array(h5Data['valid_X'])
    valid_Y = np.array(h5Data['valid_Y'])
    test_X = np.array(h5Data['test_X'])
    test_Y = np.array(h5Data['test_Y'])
    h5Data.close()

    word2index = None
    with open(cachePickle, 'rb') as data_f_pickle:
        word2index = pickle.load(data_f_pickle)

    return word2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y


if __name__ == "__main__":
    tf.app.run()