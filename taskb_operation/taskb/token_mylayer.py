import  matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skmultilearn.ext import Keras
from keras.callbacks import ModelCheckpoint
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

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('concat_1', type=int)
    parser.add_argument('rnn_dro_1', type=float)
    parser.add_argument('rnn_dro_2', type=float)
    parser.add_argument('concat_dropout_1',type=float )
    parser.add_argument('concat_dropout_2', type=float)
    parser.add_argument('concat_dropout_3', type=float)
    parser.add_argument('ker_reg_1',type=float )
    parser.add_argument('ker_reg_2', type=float)
    parser.add_argument('ker_reg_3',type=float )
    parser.add_argument('alpha_1', type=float)
    parser.add_argument('alpha_2', type=float)
    parser.add_argument('alpha_3', type=float)
    parser.add_argument('alpha_4', type=float)
    #     parser.add_argument('Binary_act_1_2', )
#     parser.add_argument('monitor', )
# # #
#     parser.add_argument('n_neighbors',type=int)
#     parser.add_argument('weights',)
#     parser.add_argument('leaf_size',type=int)
#     parser.add_argument('p',type=int)
    args = parser.parse_args()
    return  args
args = args()
parameter = pd.DataFrame({"ker_reg_1":[args.ker_reg_1],"ker_reg_2":[args.ker_reg_2],"ker_reg_3":[args.ker_reg_3],
                          })
print(parameter)


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def Attention_model(config):
    K.set_image_data_format('channels_last')
    image_input = Input(shape=(224, 224, 3), dtype='float32')
    #
    x = Conv2D(64, (5, 5), activation='relu')(image_input)
    K.set_image_data_format('channels_last')
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (5, 5), activation='relu',kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(6, (3, 3), activation='relu',kernel_regularizer=keras.regularizers.l2(args.ker_reg_1))(x)
    fla = Flatten()(x)
    image_concat = BatchNormalization()(fla)
    # image_input = Input(shape=(2048,))
    # x = keras.layers.BatchNormalization()(image_input)
    # x = Dropout(0.5)(x)
    # image_concat = Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l2(args.ker_reg_1))(x)
    # image_concat = BatchNormalization()(image_concat)
    text_input = keras.Input(shape=(config.token_number, config.token_feature_vector))
    text_bit = BatchNormalization(axis=-1)(text_input)

    enc = Bidirectional(LSTM(300, dropout=args.rnn_dro_1, return_sequences=True))(text_bit)
    enc = Bidirectional(LSTM(300, dropout=args.rnn_dro_1, return_sequences=True))(text_bit)
    enc = LSTM(300, dropout=args.rnn_dro_2, return_sequences=True,kernel_regularizer=keras.regularizers.l2(args.ker_reg_2))(enc)
    att = AttentionM()(enc)
    att = BatchNormalization()(att)
    concat = keras.layers.concatenate([image_concat,att],axis=1)
    concat_dropout_1 = BatchNormalization()(concat)
    concat_dropout_1 = Dropout(args.concat_dropout_1)(concat_dropout_1)

    concat_Dense_1 = Dense(args.concat_1,activation="relu",kernel_regularizer=keras.regularizers.l1(args.ker_reg_3))(concat_dropout_1)

    concat_dropout_2 = Dropout(args.concat_dropout_2)(concat_Dense_1)
    bitch_2 = keras.layers.BatchNormalization()(concat_dropout_2)
    dense2 = Dense(32, activation="relu")(bitch_2)

    dense3 = Dense(32, activation="relu")(bitch_2)

    dense4 = Dense(32, activation="relu")(bitch_2)

    dense5 = Dense(64, activation="relu")(bitch_2)

    Humour_Dense = Dense(1, activation='sigmoid', name='Humour')(dense2)
    Sarcasm_Dense = Dense(1, activation='sigmoid', name='Sarcasm', )(dense3)
    Offensive_Dense = Dense(1, activation='sigmoid', name='Offensive', )(dense4)
    motivational_Dense = Dense(1, activation='sigmoid', name='motivational', )(dense5)
    model = keras.Model([image_input, text_input], [Humour_Dense, Sarcasm_Dense, Offensive_Dense, motivational_Dense])
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(), loss=keras.losses.binary_crossentropy, metrics=['acc'])
    return  model


def Lstm_Model(config):
    alpha_1 = args.alpha_1
    alpha_2 = args.alpha_2
    alpha_3 = args.alpha_3
    alpha_4 = args.alpha_4
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(50, (5, 5), activation='relu')(image_input)
    K.set_image_data_format('channels_last')
    x = MaxPooling2D()(x)
    x = Conv2D(25, (5, 5), activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(25, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(15, (4, 4), activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(6, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(args.ker_reg_1))(x)
    fla = Flatten()(x)
    image_concat = BatchNormalization()(fla)
    # image_input = Input(shape=(2048,))
    # x = keras.layers.BatchNormalization()(image_input)
    # x  = Dropout(0.5)(x)
    # image_concat =Dense(300,activation="relu",kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    # image_concat = BatchNormalization()(image_concat)
    text_input = keras.Input(shape=(config.token_number, config.token_feature_vector,))

    enc = Bidirectional(LSTM(300, dropout=args.rnn_dro_1, return_sequences=True))(text_input)
    enc = Bidirectional(LSTM(300, dropout=args.rnn_dro_1, return_sequences=True))(enc)
    enc = LSTM(300, dropout=args.rnn_dro_2, kernel_regularizer=keras.regularizers.l2(args.ker_reg_2))(enc)
    text_bit = BatchNormalization()(enc)
    concat = keras.layers.concatenate([image_concat, text_bit], axis=1)
    concat_dropout_1 = Dropout(args.concat_dropout_1)(concat)
    concat_dropout_1 = BatchNormalization()(concat_dropout_1)
    concat_Dense_1 = Dense(args.concat_1, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(args.ker_reg_3))(concat_dropout_1)

    concat_dropout_2 = Dropout(args.concat_dropout_2)(concat_Dense_1)
    bitch_2 = keras.layers.BatchNormalization()(concat_dropout_2)
    dense2 = Dense(32, activation="relu")(bitch_2)

    dense3 = Dense(32, activation="relu")(bitch_2)

    dense4 = Dense(32, activation="relu")(bitch_2)

    dense5 = Dense(32, activation="relu")(bitch_2)

    Humour_Dense = Dense(1, activation='sigmoid', name='Humour')(dense2)
    Sarcasm_Dense = Dense(1, activation='sigmoid', name='Sarcasm', )(dense3)
    Offensive_Dense = Dense(1, activation='sigmoid', name='Offensive', )(dense4)
    motivational_Dense = Dense(1, activation='sigmoid', name='motivational', )(dense5)
    model = keras.Model([image_input, text_input], [Humour_Dense, Sarcasm_Dense, Offensive_Dense, motivational_Dense])
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(), loss=binary_focal_loss,metrics=['acc'])
    return model


def gruModel(config):
    """Constructs the architecture of the modelEMOTICONS_TOKEN[list_str[index]]
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    image_input = Input(shape=(224, 224, 3))
    x = keras.layers.BatchNormalization()(image_input)
    x = Conv2D(64, (5, 5), activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(32, (4, 4), activation="relu")(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (4, 4), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(6, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(args.ker_reg_1))(x)
    fla = Flatten()(x)

    image_concat = BatchNormalization()(fla)
    # x = Dropout(0.5)(x)
    # image_concat = Dense(600, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-2))(x)
    text_input = keras.Input(shape=(config.token_number, config.token_feature_vector,))
    enc = BatchNormalization(axis=-1)(text_input)

    enc = Bidirectional(GRU(300, dropout=args.rnn_dro_1, return_sequences=True))(enc)
    enc = Bidirectional(GRU(300, dropout=args.rnn_dro_1, return_sequences=True))(enc)
    enc = GRU(300, dropout=args.rnn_dro_2, kernel_regularizer=keras.regularizers.l2(args.ker_reg_2))(enc)
    text_bit = BatchNormalization()(enc)
    concat = keras.layers.concatenate([image_concat, text_bit], axis=1)
    concat_dropout_1 = BatchNormalization()(concat)
    concat_dropout_1 = Dropout(args.concat_dropout_1)(concat_dropout_1)
    concat_Dense_1 = Dense(args.concat_1, activation="relu",
                           kernel_regularizer=keras.regularizers.l1(args.ker_reg_3))(
        concat_dropout_1)
    concat_dropout_2 = Dropout(args.concat_dropout_2)(concat_Dense_1)
    bitch_2 = keras.layers.BatchNormalization()(concat_dropout_2)
    dense2 = Dense(32, activation="relu")(bitch_2)
    #
    dense3 = Dense(32, activation="relu")(bitch_2)

    dense4 = Dense(32, activation="relu")(bitch_2)

    dense5 = Dense(32, activation="relu")(bitch_2)


    Humour_Dense = Dense(1, activation='sigmoid', name='Humour')(dense2)
    Sarcasm_Dense = Dense(1, activation='sigmoid', name='Sarcasm', )(dense3)
    Offensive_Dense = Dense(1, activation='sigmoid', name='Offensive', )(dense4)
    motivational_Dense = Dense(1, activation='sigmoid', name='motivational', )(dense5)
    model = keras.Model([image_input, text_input], [Humour_Dense,Sarcasm_Dense,Offensive_Dense,motivational_Dense])
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(), loss=keras.losses.binary_crossentropy, metrics=['acc'])
    return model


def Sentence_BinaryRelevance_Model(k,config,xTrain_picture,xTrain_sentence,yTrain,xVal_picture, xVal_sentence,yVal,x_test_picture,x_test_sentence):
    Train_Humour_label = np.array(yTrain[:, 0])
    Train_Sarcasm_label = np.array(yTrain[:, 1])
    Train_offensive_label = np.array(yTrain[:, 2])
    Train_motivational_label = np.array(yTrain[:, 3])

    Val_Humour_label = np.array(yVal[:, 0])
    Val_Sarcasm_label = np.array(yVal[:, 1])
    Val_offensive_label = np.array(yVal[:, 2])
    Val_motivational_label = np.array(yVal[:, 3])

    filepath ="./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_Humour")
    callbacks_list_Humour= get_checkpoint(filepath,"val_acc")
    clf_0 =create_model_multiclass_Humour(config)
    history0 = clf_0.fit(x=xTrain_sentence, y=Train_Humour_label,
                         batch_size=128, epochs=70,
                         validation_data=(xVal_sentence, Val_Humour_label),
                         callbacks=callbacks_list_Humour)
    clf_0 = keras.models.load_model(filepath,custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
    y_pred_0 = clf_0.predict(x_test_sentence)
    # #
    val_pred_0 = clf_0.predict(xVal_sentence)

    filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_Sarcasm")
    callbacks_list_Sarcasm = get_checkpoint(filepath, "val_acc")
    clf_1= create_model_multiclass_Sarcasm(config)
    history1 = clf_1.fit(xTrain_sentence, Train_Sarcasm_label,
                         batch_size=128, epochs=70,
                         validation_data=(xVal_sentence, Val_Sarcasm_label),
                         callbacks=callbacks_list_Sarcasm)
    clf_1 = keras.models.load_model(filepath, custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
    y_pred_1 = clf_1.predict(x_test_sentence)
    val_pred_1 = clf_1.predict(xVal_sentence)
    filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_offensive")
    callbacks_list_offensive= get_checkpoint(filepath,"val_acc")
    clf_2= create_model_multiclass_offensive(config)
    history2 = clf_2.fit(xTrain_sentence, Train_offensive_label,
              batch_size=128, epochs=70, validation_data=(xVal_sentence,Val_offensive_label),
              callbacks=callbacks_list_offensive)
    clf_2 = keras.models.load_model(filepath)
    y_pred_2 = clf_2.predict(x_test_sentence)
    val_pred_2 = clf_2.predict(xVal_sentence)
    filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_motivational")
    callbacks_list_offensive = get_checkpoint(filepath, "val_acc")
    clf_3 = create_model_multiclass_motivational(config)
    history3 = clf_3.fit(xTrain_sentence, Train_motivational_label,
                         batch_size=128, epochs=70, validation_data=(xVal_sentence, Val_motivational_label),
                         callbacks=callbacks_list_offensive)
    clf_3 = keras.models.load_model(filepath, custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
    y_pred_3 = clf_3.predict(x_test_sentence)
    val_pred_3 = clf_3.predict(xVal_sentence)
    # print(y_pred_1.shape)]
    y_pred_0 = np.array(y_pred_0)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)
    y_pred_3 = np.array(y_pred_3)

    val_pred_0 = np.array(val_pred_0)
    val_pred_1 = np.array(val_pred_1)
    val_pred_2 = np.array(val_pred_2)
    val_pred_3 = np.array(val_pred_3)
    return  y_pred_0,y_pred_1,y_pred_2,y_pred_3,val_pred_0,val_pred_1,val_pred_2,val_pred_3


def create_model_multiclass_Humour(config):
    print(config.Binary_act_1_1)
    text_input = Input(shape=(768,))
    text_bat= keras.layers.BatchNormalization()(text_input)
    print(config.Binary_hidden1_1)
    x = Dense(config.Binary_hidden1_1, activation=config.Binary_act_1_1,kernel_regularizer=keras.regularizers.l2(1e-2))(text_bat)
    x = Dropout(config.Binary_dro_1_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden1_2, activation=config.Binary_act_1_2)(x)
    x = Dropout(config.Binary_dro_1_2)(x)
    out_puts =Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model(text_input, out_puts)
    model.compile(loss=[binary_focal_loss()], optimizer=keras.optimizers.adam(0.0001), metrics=['accuracy'])
    return model

def create_model_multiclass_Sarcasm(config):
    # create model

    text_input = Input(shape=(768,))
    text_bat = keras.layers.BatchNormalization()(text_input)
    x = Dense(config.Binary_hidden2_1, activation=config.Binary_act_2_1,
              kernel_regularizer=keras.regularizers.l2(1e-2))(text_bat)
    x = Dropout(config.Binary_dro_2_1)(x)
    x =keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden2_2, activation=config.Binary_act_2_2)(x)
    x = Dropout(config.Binary_dro_2_2)(x)
    out_puts = Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model(text_input, out_puts)
    model.compile(loss=binary_focal_loss(), optimizer=keras.optimizers.adam(0.0001), metrics=['accuracy'])
    model.summary()
    return model

def create_model_multiclass_offensive(config):
    # create model
    text_input = Input(shape=(768,))
    text_bat = keras.layers.BatchNormalization()(text_input)
    # concat = keras.layers.concatenate([image_bat,text_bat],axis=1)
    x = Dense(config.Binary_hidden3_1, activation=config.Binary_act_3_1,
              kernel_regularizer=keras.regularizers.l2(1e-3))(text_bat)
    x = Dropout(config.Binary_dro_3_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden3_2, activation=config.Binary_act_3_2)(x)
    x = Dropout(config.Binary_dro_3_2)(x)
    out_puts = Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model( text_input, out_puts)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.adam(0.001), metrics=['accuracy'])
    model.summary()

    return model



def create_model_multiclass_motivational(config):
    text_input = Input(shape=(768,))
    text_bat = keras.layers.BatchNormalization()(text_input)
    x = Dense(config.Binary_hidden4_1, activation=config.Binary_act_3_1,
              kernel_regularizer=keras.regularizers.l2(1e-3))(text_bat)
    x = Dropout(config.Binary_dro_4_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden4_2, activation=config.Binary_act_4_2)(x)
    x = Dropout(config.Binary_dro_4_2)(x)
    out_puts = Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model( text_input, out_puts)
    model.compile(loss=[binary_focal_loss()], optimizer=keras.optimizers.adam(0.001), metrics=['accuracy'])
    model.summary()

    return model





def Picture_BinaryRelevance_Model(k,config,xTrain_picture,xTrain_sentence,yTrain,xVal_picture, xVal_sentence,yVal,x_test_picture,x_test_sentence):
    Train_Humour_label = np.array(yTrain[:, 0])
    Train_Sarcasm_label = np.array(yTrain[:, 1])
    Train_offensive_label = np.array(yTrain[:, 2])
    Train_motivational_label = np.array(yTrain[:, 3])

    Val_Humour_label = np.array(yVal[:, 0])
    Val_Sarcasm_label = np.array(yVal[:, 1])
    Val_offensive_label = np.array(yVal[:, 2])
    Val_motivational_label = np.array(yVal[:, 3])

    filepath ="./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_Humour")
    callbacks_list_Humour= get_checkpoint(filepath,"val_acc")
    clf_0 =picture_multiclass_Humour(config)
    history0 = clf_0.fit(x=[xTrain_picture,xTrain_sentence], y=Train_Humour_label,
                         batch_size=128, epochs=70,
                         validation_data=([xVal_picture,xVal_sentence], Val_Humour_label),
                         callbacks=callbacks_list_Humour)
    clf_0 = keras.models.load_model(filepath,custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
    y_pred_0 = clf_0.predict([x_test_picture,x_test_sentence])
    # #
    val_pred_0 = clf_0.predict([xVal_picture,xVal_sentence])

    filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_Sarcasm")
    callbacks_list_Sarcasm = get_checkpoint(filepath, "val_acc")
    clf_1= picture_multiclass_Sarcasm(config)
    history1 = clf_1.fit([xTrain_picture,xTrain_sentence], Train_Sarcasm_label,
                         batch_size=128, epochs=70,
                         validation_data=([xVal_picture,xVal_sentence], Val_Sarcasm_label),
                         callbacks=callbacks_list_Sarcasm)
    clf_1 = keras.models.load_model(filepath, custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
    y_pred_1 = clf_1.predict([x_test_picture,x_test_sentence])

    val_pred_1 = clf_1.predict([xVal_picture, xVal_sentence])

    filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_offensive")
    callbacks_list_offensive= get_checkpoint(filepath,"val_loss")
    clf_2= picture_multiclass_offensive(config)
    history2 = clf_2.fit([xTrain_picture,xTrain_sentence], Train_offensive_label,
              batch_size=128, epochs=70, validation_data=([xVal_picture,xVal_sentence],Val_offensive_label),
              callbacks=callbacks_list_offensive)
    clf_2 = keras.models.load_model(filepath)
    y_pred_2 = clf_2.predict([x_test_picture,x_test_sentence])

    val_pred_2 = clf_2.predict([xVal_picture, xVal_sentence])

    filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, "create_model_multiclass_motivational")
    callbacks_list_offensive = get_checkpoint(filepath, "val_acc")
    clf_3 = picture_multiclass_motivational(config)
    history3 = clf_3.fit([xTrain_picture,xTrain_sentence], Train_motivational_label,
                         batch_size=128, epochs=70, validation_data=([xVal_picture,xVal_sentence], Val_motivational_label),
                         callbacks=callbacks_list_offensive)
    clf_3 = keras.models.load_model(filepath, custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})

    y_pred_3 = clf_3.predict([x_test_picture,x_test_sentence])
    val_pred_3 = clf_3.predict([xVal_picture, xVal_sentence])
    # print(y_pred_1.shape)]
    y_pred_0 = np.array(y_pred_0)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)
    y_pred_3 = np.array(y_pred_3)
    val_pred_0 = np.array(val_pred_0)
    val_pred_1 = np.array(val_pred_1)
    val_pred_2 = np.array(val_pred_2)
    val_pred_3 = np.array(val_pred_3)
    return  y_pred_0,y_pred_1,y_pred_2,y_pred_3,val_pred_0,val_pred_1,val_pred_2,val_pred_3


def picture_multiclass_Humour(config):
    picture_input = Input(shape=(768,))
    picture_bat = keras.layers.BatchNormalization()(picture_input)
    text_input = Input(shape=(768,))
    text_bat= keras.layers.BatchNormalization()(text_input)

    concat = keras.layers.concatenate([picture_bat,text_bat],axis=1)
    concat_bat = keras.layers.BatchNormalization()(concat)
    x = Dense(config.Binary_hidden1_1, activation=config.Binary_act_1_1,kernel_regularizer=keras.regularizers.l2(1e-2))(concat_bat)
    x = Dropout(config.Binary_dro_1_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden1_2, activation=config.Binary_act_1_2)(x)
    x = Dropout(config.Binary_dro_1_2)(x)
    out_puts =Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model([picture_input,text_input], out_puts)
    model.compile(loss=[binary_focal_loss()], optimizer=keras.optimizers.adam(0.0001), metrics=['accuracy'])
    return model

def picture_multiclass_Sarcasm(config):
    # create model

    picture_input = Input(shape=(768,))
    picture_bat = keras.layers.BatchNormalization()(picture_input)
    text_input = Input(shape=(768,))
    text_bat = keras.layers.BatchNormalization()(text_input)

    concat = keras.layers.concatenate([picture_bat, text_bat], axis=1)
    concat_bat = keras.layers.BatchNormalization()(concat)
    x = Dense(config.Binary_hidden2_1, activation=config.Binary_act_2_1,
              kernel_regularizer=keras.regularizers.l2(1e-2))(concat_bat)
    x = Dropout(config.Binary_dro_2_1)(x)
    x =keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden2_2, activation=config.Binary_act_2_2)(x)
    x = Dropout(config.Binary_dro_2_2)(x)
    out_puts = Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model([picture_input,text_input], out_puts)
    model.compile(loss=binary_focal_loss(), optimizer=keras.optimizers.adam(0.0001), metrics=['accuracy'])
    model.summary()
    return model

def picture_multiclass_offensive(config):
    # create model
    picture_input = Input(shape=(768,))
    picture_bat = keras.layers.BatchNormalization()(picture_input)
    text_input = Input(shape=(768,))
    text_bat = keras.layers.BatchNormalization()(text_input)
    concat = keras.layers.concatenate([picture_bat, text_bat], axis=1)
    concat_bat = keras.layers.BatchNormalization()(concat)
    x = Dense(config.Binary_hidden3_1, activation=config.Binary_act_3_1,
              kernel_regularizer=keras.regularizers.l2(1e-2))(concat_bat)
    x = Dropout(config.Binary_dro_3_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden3_2, activation=config.Binary_act_3_2)(x)
    x = Dropout(config.Binary_dro_3_2)(x)
    out_puts = Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model( [picture_input,text_input], out_puts)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.adam(0.001), metrics=['accuracy'])
    model.summary()
    return model



def picture_multiclass_motivational(config):
    picture_input = Input(shape=(768,))
    picture_bat = keras.layers.BatchNormalization()(picture_input)
    text_input = Input(shape=(768,))
    text_bat = keras.layers.BatchNormalization()(text_input)
    concat = keras.layers.concatenate([picture_bat, text_bat], axis=1)
    concat_bat = keras.layers.BatchNormalization()(concat)
    x = Dense(config.Binary_hidden4_1, activation=config.Binary_act_3_1,
              kernel_regularizer=keras.regularizers.l2(1e-3))(concat_bat)
    x = Dropout(config.Binary_dro_4_1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(config.Binary_hidden4_2, activation=config.Binary_act_4_2)(x)
    x = Dropout(config.Binary_dro_4_2)(x)
    out_puts = Dense(1, activation='sigmoid')(x)
    # Compile model
    model = keras.Model( [picture_input,text_input], out_puts)
    model.compile(loss=[binary_focal_loss(alpha=0.7)], optimizer=keras.optimizers.adam(0.001), metrics=['accuracy'])
    model.summary()

    return model