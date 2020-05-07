import  matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skmultilearn.ext import Keras
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import  sklearn
from sklearn.neighbors import  KNeighborsClassifier
from skmultilearn.adapt import BRkNNaClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense, Dropout, Embedding, CuDNNGRU, Bidirectional, GRU, Input, Flatten, SpatialDropout1D, LSTM
from sklearn.svm import SVC
from keras.models import Model
from keras.layers import Dense,Dropout
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential,Input
import keras
import pandas as pd
from sklearn.svm import SVC
from tensorflow.python.ops import array_ops
from keras import backend as K
import matplotlib.pyplot as plt
import  numpy as np
import tensorflow as tf
from  collections import Counter

from keras.utils import multi_gpu_model

from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
from scipy.sparse import *
from Attention_layer import AttentionM, cnnModel
import  time
import argparse
num_classes = 2
#smooth 参数防止分母为0
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def get_checkpoint(filepath,my_monitor):
    checkpoint = ModelCheckpoint(filepath, monitor=my_monitor, verbose=1, save_best_only=True,
                                 mode='auto')
    callbacks_list = [checkpoint]
    return callbacks_list

def get_acc_loss_img(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation acc values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Acc Humour')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./picture/picture+{0}.jpg'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    try:
        plt.show()
    except Exception as e:
        print("")

# def args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('concat_1', type=int)
#     parser.add_argument('rnn_dro_1', type=float)
#     parser.add_argument('rnn_dro_2', type=float)
#     parser.add_argument('concat_dropout_1',type=float )
#     parser.add_argument('concat_dropout_2', type=float)
#     parser.add_argument('concat_dropout_3', type=float)
#     parser.add_argument('ker_reg_1',type=float )
#     parser.add_argument('ker_reg_2', type=float)
#     parser.add_argument('ker_reg_3',type=float )
# # # # # #
# #      parser.add_argument('Binary_hidden1_1',type=int)
# #      parser.add_argument('Binary_hidden1_2',type=int)
# #      parser.add_argument('Binary_dro_1_1',type=float)
# #      parser.add_argument('Binary_dro_1_1',type=float)
#     args = parser.parse_args()
#     return  args
# args = args()
# # # parameter = pd.DataFrame({"concat_1":[args.concat_1],"rnn_dro_1":[args.rnn_dro_1],"rnn_dro_2":[args.rnn_dro_2],
# # #                           "concat_dropout_1":[args.concat_dropout_1],"concat_dropout_2":[args.concat_dropout_2],"concat_dropout_3":[args.concat_dropout_3]})
# # # print(parameter)
# #
# # parameter = pd.DataFrame({"ker_reg_1":[args.ker_reg_1],"ker_reg_2":[args.ker_reg_2],"ker_reg_3":[args.ker_reg_3],
# #                           })
# # print(parameter)


def focal_loss( gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    classes_num =[2,5,10]
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed


def recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def f1(y_true, y_pred):
  def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
  return f1

def Attention_Model(config):
    image_input = Input(shape=(224, 224, 3))
    x = keras.layers.BatchNormalization()(image_input)
    x = Conv2D(64, (4, 4), activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (2, 2), activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (2, 2), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = keras.layers.Conv2D(10, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(config.ker_reg_1))(x)
    fla = Flatten()(x)
    image_concat = BatchNormalization()(fla)
    # image_input = Input(shape=(2048,))
    # x = keras.layers.BatchNormalization()(image_input)
    # x = Dropout(0.5)(x)
    # image_concat = Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l2(args.ker_reg_1))(x)
    # image_concat = BatchNormalization()(image_concat)
    text_input = keras.Input(shape=(config.token_number,config.token_feature_vector,))
    # text_input = keras.layers.Masking(mask_value=0.0,input_shape=(config.token_number,config.token_feature_vector,))
    # text_bit = BatchNormalization(axis=-1)(text_input)

    enc = Bidirectional(LSTM(300, dropout=config.rnn_dro_1, return_sequences=True))(text_input)
    enc = Bidirectional(LSTM(300, dropout=config.rnn_dro_1, return_sequences=True))(enc)
    enc = LSTM(160, dropout=config.rnn_dro_2, return_sequences=True,kernel_regularizer=keras.regularizers.l2(config.ker_reg_2))(enc)
    att = AttentionM()(enc)
    att = BatchNormalization()(att)
    concat = keras.layers.concatenate([image_concat,att],axis=1)
    concat_dropout_1 = BatchNormalization()(concat)
    concat_dropout_1 = Dropout(config.concat_dropout_1)(concat_dropout_1)

    concat_Dense_1 = Dense(config.concat_1,activation="relu",kernel_regularizer=keras.regularizers.l1(config.ker_reg_3))(concat_dropout_1)

    concat_dropout_2 = Dropout(config.concat_dropout_2)(concat_Dense_1)

    dense2 = Dense(64,activation="relu",name="Dense_2")(concat_dropout_2)
    concat_dropout_3 =Dropout(config.concat_dropout_3)(dense2)
    dense3 = Dense(3, activation='softmax',)(concat_dropout_3)

    model = keras.Model([image_input, text_input], dense3)
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(), loss=keras.losses.categorical_crossentropy, metrics=['acc'])
    return  model


def Lstm_Model(config):
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(50, (5, 5), activation='relu')(image_input)
    K.set_image_data_format('channels_last')
    x = MaxPooling2D()(x)
    x = Conv2D(25, (5, 5), activation='relu',kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(25, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(15, (4, 4), activation='relu',kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(6, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(config.ker_reg_1))(x)
    fla = Flatten()(x)
    image_concat = BatchNormalization()(fla)
    # image_input = Input(shape=(2048,))
    # x = keras.layers.BatchNormalization()(image_input)
    # x  = Dropout(0.5)(x)
    # image_concat =Dense(300,activation="relu",kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    # image_concat = BatchNormalization()(image_concat)
    text_input = keras.Input(shape=(config.token_number,config.token_feature_vector,))

    enc = Bidirectional(LSTM(300, dropout=config.rnn_dro_1, return_sequences=True))(text_input)
    enc = Bidirectional(LSTM(300, dropout=config.rnn_dro_1, return_sequences=True))(enc)
    enc = Bidirectional(LSTM(300, dropout=config.rnn_dro_2,kernel_regularizer=keras.regularizers.l2(config.ker_reg_2)))(enc)
    text_bit = BatchNormalization()(enc)
    concat = keras.layers.concatenate([image_concat,text_bit], axis=1)
    concat_dropout_1 = Dropout(config.concat_dropout_1)(concat)
    concat_dropout_1 = BatchNormalization()(concat_dropout_1)
    concat_Dense_1 = Dense(config.concat_1, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(config.ker_reg_3))(concat_dropout_1)

    concat_dropout_2 = Dropout(config.concat_dropout_2)(concat_Dense_1)

    concat_Dense_2 = Dense(64,activation="relu",name="Dense_2")(concat_dropout_2)

    concat_dropout_3 = Dropout(config.concat_dropout_3)(concat_Dense_2)
    dense3 = Dense(3, activation='softmax', )(concat_dropout_3)

    model = keras.Model([image_input, text_input],dense3)
    model.summary()
    model.compile(optimizer=keras.optimizers.adadelta(), loss=keras.losses.categorical_crossentropy, metrics=['acc'])
    return model


def gruModel(config):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    # image_input = Input(shape=(2048,))
    # x = keras.layers.BatchNormalization()(image_input)
    image_input = Input(shape=(224, 224, 3))
    x = keras.layers.BatchNormalization()(image_input)
    x = Conv2D(64, (4, 4), activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (2, 2), activation="relu",kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (2, 2), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = keras.layers.Conv2D(10, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(config.ker_reg_1))(x)
    fla = Flatten()(x)

    image_concat = BatchNormalization()(fla)
    # x = Dropout(0.5)(x)
    # image_concat = Dense(600, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-2))(x)
    text_input = keras.Input(shape=(config.token_number, config.token_feature_vector,))
    enc = BatchNormalization(axis=-1)(text_input)

    enc = Bidirectional(GRU(300, dropout=config.rnn_dro_1, return_sequences=True))(enc)
    enc = Bidirectional(GRU(300, dropout=config.rnn_dro_1, return_sequences=True))(enc)
    enc = GRU(160, dropout=config.rnn_dro_2,kernel_regularizer=keras.regularizers.l2(config.ker_reg_2))(enc)
    text_bit = BatchNormalization()(enc)
    concat = keras.layers.concatenate([image_concat, text_bit], axis=1)
    concat_dropout_1 = Dropout(config.concat_dropout_1)(concat)
    concat_dropout_1 = BatchNormalization()(concat_dropout_1)
    concat_Dense_1 = Dense(config.concat_1, activation="relu", kernel_regularizer=keras.regularizers.l1(config.ker_reg_3))(
        concat_dropout_1)

    concat_dropout_2 = Dropout(config.concat_dropout_2)(concat_Dense_1)

    concat_Dense_2 = Dense(64, activation="relu",name="Dense_2")(concat_dropout_2)
    concat_dropout_2 = Dropout(config.concat_dropout_3)(concat_Dense_2)
    dense3 = Dense(3, activation='softmax', )(concat_dropout_2)

    model = keras.Model([image_input, text_input], dense3)
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(), loss=keras.losses.binary_crossentropy, metrics=['acc'])
    return model

def Dense_Model(config):
    text_input = Input(shape=(768,))

    text_bit = keras.layers.BatchNormalization()(text_input)

    image_input = Input(shape=(768,))

    image_bit = keras.layers.BatchNormalization()(image_input)
    x = keras.layers.concatenate([image_bit, text_bit],axis=1)
    x = Dense(config.Binary_hidden1_1, activation="relu",
              kernel_regularizer=keras.regularizers.l2(1e-2))(x)
    x = Dropout(config.Binary_dro_1_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden1_2, activation="relu")(x)
    x = Dropout(config.Binary_dro_1_2)(x)
    out_puts = Dense(3, activation='softmax')(x)
    # Compile model
    model = keras.Model([image_input,text_input], out_puts)
    model.compile(loss=[keras.losses.categorical_crossentropy], optimizer=keras.optimizers.adam(0.0001), metrics=['accuracy'])
    return model