import tensorflow as tf
from textCNN_model import TextCNN
import os
import util
import textCNN_train

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.flags.DEFINE_float("decay_rate", 1, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.flags.DEFINE_integer("sentence_len", 30, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.flags.DEFINE_boolean("is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
tf.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
filter_sizes = [5, 6, 7]


#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    word2index, _, _, _, _, testX, testY = textCNN_train.load_data(util.dataPath + 'TrainTest.h5py', util.dataPath + 'word2index.pickle')
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:", vocab_size)

    num_examples, FLAGS.sentence_len = testX.shape
    print("num_examples of test:", num_examples, " ;sentence_len:", FLAGS.sentence_len)

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
            print('no checkpoint file in ', util.modelPath)

        test_loss, f1_score, f1_micro, f1_macro = textCNN_train.do_eval(sess, textCNN, testX, testY, 15, False)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (test_loss, f1_score, f1_micro, f1_macro))


if __name__ == "__main__":
    tf.app.run()