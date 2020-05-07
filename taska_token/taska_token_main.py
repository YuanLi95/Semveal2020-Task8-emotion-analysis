from __future__ import print_function
import h5py
import  numpy as np
from Taska_My_Config import Config
from Attention_layer import AttentionM, cnnModel
import  pandas as pd
import  matplotlib
matplotlib.use('Agg')
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
import  re
from taska_token_mylayer import Attention_Model,Lstm_Model,focal_loss,gruModel,focal_loss,Dense_Model
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import  random
import  pickle
import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from keras import backend as K
import tensorflow as tf
from collections import Counter
from keras import layers
import  pickle

from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D,Dense, Dropout, Activation, Flatten,concatenate,normalization
from keras.models import Sequential
from keras.models import save_model

from  keras.models import load_model
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from  collections import Counter
from keras.callbacks import EarlyStopping
import  tensorflow as tf
from torchvision import models, transforms
from keras.utils import plot_model
from sklearn.svm import SVC
from scipy import sparse
from  keras.wrappers.scikit_learn import  KerasClassifier
import os
import time
import argparse
from keras.callbacks import ModelCheckpoint
import  pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from ast import literal_eval
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
source_path = "../../change_size/picture_data7000/"


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
# # # # #
#      parser.add_argument('Binary_hidden1_1',type=int)
#      parser.add_argument('Binary_hidden1_2',type=int)
#      parser.add_argument('Binary_dro_1_1',type=float)
#      parser.add_argument('Binary_dro_1_1',type=float)
    args = parser.parse_args()
    return  args
args = args()

num_classes = 3
batch_size = 32
NUM_EPOCHS = 40
NUM_FOLDS =5
sent_cont_pic_2 = pd.DataFrame(columns=["contact_vector","label"])
def multi_category_focal_loss(y_true, y_pred):
    epsilon = 1.e-7
    gamma = 2
    # alpha = tf.constant([[2],[1],[1],[1],[1]], dtype=tf.float32)
    alpha = tf.constant([[2], [1], [1]], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss

#
#获得图片对应的numpy
def get_picture_numpy(image_names):
    picture_list = []
    for index, row in image_names.iterrows():
        image_name = row["Image_name"]
        img_path = source_path + image_name
        image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        pic_vector = image / 255
        picture_list.append(pic_vector)
    picture_list = np.vstack(picture_list)
    return  picture_list

def get_test_picture_numpy(image_names):
    picture_list = []
    for index, row in image_names.iterrows():
        image_name = row["Image_name"]
        img_path = source_path + image_name
        image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        pic_vector = image / 255
        picture_list.append(pic_vector)
    picture_list = np.vstack(picture_list)
    return  picture_list

#将非0的label转换为1
def get_data(config):
    # 训练集的两个分量
    #如果是二维关系就用sentence_embedding and picture_vertor
    if config.token_type=="elmo":
       # 获得句向量和图片向量
        trian_sentence_path = "../../data/train_data/sentence_data/elmo_token96_train.pickle"
        x_train_sentence = pickle.load(open(trian_sentence_path, 'rb'))
        test_sentence_path = "../../data/test_data/sentence_data/elmo_token96_test.pickle"
        x_test_sentence = pickle.load(open(test_sentence_path, 'rb'))
        x_train_picture = pd.read_csv("../../data/train_data/sentence_data/data_have_sentence.csv", header=0, delimiter=",")
        x_train_picture = get_picture_numpy(x_train_picture)
        print(x_train_picture.shape)
        # print(x_train_picture.shape)
        # # x_val_picture = pd.read_csv("../../data/val_data/val_image_label.csv", delimiter="\t", header=None)
        # # x_val_picture = get_picture_numpy(x_val_picture)

        x_test_picture = pd.read_csv("../../data/test_data/sentence_data/test_data_all.csv", delimiter=",", header=0)
        x_test_picture = get_picture_numpy(x_test_picture)
    elif config.token_type=="bert":
    #获得句向量和图片向量
        trian_sentence_path = "../../data/train_data/sentence_data/bert_token128_train_new.pickle"
        x_train_sentence = pickle.load(open(trian_sentence_path, 'rb'))

        # val_sentence_path = "../../data/val_data/sentence_data/bert_token128_val_new.pickle"
        # x_val_sentence = pickle.load(open(val_sentence_path, 'rb'))

        test_sentence_path = "../../data/test_data/sentence_data/bert_token128_test.pickle"
        x_test_sentence = pickle.load(open(test_sentence_path, 'rb'))
        x_train_picture = pd.read_csv("../../data/train_data/sentence_data/data_have_sentence.csv", header=0, delimiter=",")
        x_train_picture = get_picture_numpy(x_train_picture)
        # print(x_train_picture.shape)
        # # x_val_picture = pd.read_csv("../../data/val_data/val_image_label.csv", delimiter="\t", header=None)
        # # x_val_picture = get_picture_numpy(x_val_picture)

        x_test_picture = pd.read_csv("../../data/test_data/sentence_data/test_data_all.csv", delimiter=",", header=0)
        x_test_picture = get_picture_numpy(x_test_picture)
        # x_test_sentence=np.zeros((1,2))
    else: #获得sentence_embedding
        trian_sentence_path = "../../data/train_data/sentence_data/bert_sentence_train_new.pickle"
        x_train_sentence = pickle.load(open(trian_sentence_path, 'rb'))

        x_train_picture_path = "../../data/train_data/picture_data/train_vector_768.pickle"
        x_train_picture = pickle.load(open(x_train_picture_path, 'rb'))
        x_train_picture = np.array(x_train_picture)

        test_sentence_path = "../../data/test_data/sentence_data/bert_sentence_test.pickle"
        x_test_sentence = pickle.load(open(test_sentence_path, 'rb'))

        x_test_picture_path = "../../data/test_data/picture_data/test_vector_768.pickle"
        x_test_picture = pickle.load(open(x_test_picture_path, 'rb'))
        x_test_picture = np.array(x_test_picture)


    #获得图片的numpy
    # x_test_picture=np.zeros((1,2))
    # x_train_picture_path = "../../data/train_data/picture_data/train_nofc_picture.pickle"
    # x_train_picture = pickle.load(open(x_train_picture_path, 'rb'))
    # x_train_picture = np.array(x_train_picture)
    # x_train_picture = np.squeeze(x_train_picture)
    # x_val_picture_path = "../../data/val_data/picture_data/val_nofc_picture.pickle"
    # x_val_picture = pickle.load(open(x_val_picture_path, 'rb'))
    # x_val_picture = np.array(x_val_picture)
    # x_val_picture = np.squeeze(x_val_picture)

    #获得train_label val_label
    train_label = pd.read_csv("../../data/train_data/sentence_data/data_have_sentence.csv",header=0,delimiter=",")
    y_train =np.array(train_label["Overall_Sentiment"]).tolist()
    y_train = keras.utils.to_categorical(y_train,3)

    # val_label = pd.read_csv("../../data/val_data/val_image_label.csv", delimiter="\t", header=None)
    # val_label = np.array(val_label.iloc[:, 4])

    # x_val_sentence = np.array(x_val_sentence)
    x_train_sentence= np.array(x_train_sentence)
    x_test_sentence = np.array(x_test_sentence)
    print(x_test_sentence.shape)
    print(x_test_picture.shape)
    print(x_train_sentence.shape)
    print(x_train_picture.shape)
    # print(x_val_sentence.shape)
    # print(x_val_picture.shape)
    print(y_train.shape)
    # print(val_label.shape)
    print(x_test_sentence.shape)
    print(x_test_picture.shape)

    # return  x_train_picture,x_train_sentence,y_train,x_val_sentence,x_val_picture,val_label,x_test_picture,x_test_sentence
    return  x_train_picture,x_train_sentence,y_train,x_test_picture,x_test_sentence


def panda_str_to_array_vector(rev):
    new_rev = []
    i = rev
    # i= eval(i)
    # 转换后数字变成了字符，所以只能把它当成字符串用正则表达式方式来获得原始的list
    # new_i = []
    # 删除掉符号[]
    i = re.sub('[\[\]]', "", i)
    # 获得只有数字的list
    i = i.split(",")
    # 将数字由str格式转换成float格式
    new_rev = [float(num) for num in i]
    new_rev =np.array(new_rev)
    return new_rev



def get_acc_loss_img(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation acc values


    # Plot training & validation acc values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Acc ')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig('./picture/picture+{0}.jpg'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    plt.close()

#得到result文件 :
def  get_result_file(y_pred):
    pd_reslut = pd.DataFrame()
    for index in y_pred:
        if index == 0:
            i = -1
        elif index == 1:
            i = 0
        elif  index == 2:
            i = 1
        every_result = "{0}_9999_9999".format(i)
        pd_reslut = pd_reslut.append(pd.Series(every_result),ignore_index=True)
    pd_reslut.to_csv("./report/answer{0}.txt".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),index=None,header=0)
#预留test空间
def get_stacking(config,x_train_picture,x_train_sentence,y_train,x_test_picture,x_test_sentence):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num,= x_train_picture.shape[0], x_test_picture.shape[0],
    second_level_train_set = np.zeros((train_num, 1))
    test_result = np.zeros((test_num, 3))
    test_nfolds_sets = []
    # for k in range(1):
    for k in range(NUM_FOLDS):
        validationSize = int(len(x_train_sentence)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k + 1)
        #
        xTrain_sentence= np.vstack((x_train_sentence[:index1],x_train_sentence[index2:]))
        xTrain_picture = np.vstack((x_train_picture[:index1], x_train_picture[index2:]))
        yTrain = np.vstack((y_train[:index1], y_train[index2:]))
        #
        xVal_sentence = x_train_sentence[index1:index2]
        xVal_picture = x_train_picture[index1:index2]
        yVal = y_train[index1:index2]
        # xTrain_sentence = x_train_sentence
        # xTrain_picture = x_train_picture
        # yTrain = y_train
        # y_val = keras.utils.to_categorical(y_val,num_classes=3)
        # print(xTrain_picture.shape)
        # print(xTrain_sentence.shape)

        if config.model_name == "Attention_Model":
            clf = Attention_Model(config)
        if config.model_name =="Lstm_Model":
            clf = Lstm_Model(config)
        if config.model_name =="gruModel":
            clf = gruModel(config)
        if config.model_name =="Dense_Model":
            clf = Dense_Model(config)

        config.filepath = "./Model/{0}+n_fold+{1}+{2}.hdf5".format(k, config.model_name,config.token_type)
        reduce_lr =  keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, mode='auto')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(config.filepath, monitor='val_loss', mode='auto', verbose=1,
                                                                 save_best_only=True)
        early_point = EarlyStopping(monitor="val_loss", mode='auto', patience=5)
        callbacks_list = [checkpoint_callback,reduce_lr,early_point]

        y_integers = np.argmax(yTrain, axis=1)
        my_class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(my_class_weights))
        # print(d_class_weights)
        # d_class_weights = {0:7.5, 1:2.5, 2:1}
        print(d_class_weights)
        history = clf.fit([xTrain_picture ,xTrain_sentence],[yTrain]
                          ,batch_size=args.batch_size, epochs=18, validation_data=([xVal_picture,xVal_sentence],yVal),
                          callbacks=callbacks_list,verbose=1,class_weight=d_class_weights)
        #
        # history = clf.fit(xTrain_sentence,yTrain,shuffle=True,
        #                   batch_size=100, epochs=100, validation_split=0.1,class_weight= d_class_weights,
        #                   callbacks=callbacks_list, verbose=1)

        get_acc_loss_img(history)
        if config.model_name == "Attention_Model":
            # clf = load_model(config.filepath,custom_objects={'AttentionM':AttentionM,'binary_focal_loss_fixed': binary_focal_loss()})
            clf = load_model(config.filepath,custom_objects={'AttentionM':AttentionM})
        elif config.model_name=="Dense_Model":
           # clf = load_model(config.filepath,custom_objects={'focal_loss_fixed': focal_loss()})
            clf = load_model(config.filepath)
        else:
            clf = load_model(config.filepath)

        y_pred = clf.predict([x_test_picture,x_test_sentence])
        y_val_pred = clf.predict([xVal_picture,xVal_sentence])
        # y_pred = clf.predict(x_test_sentence)
        # y_val_pred = clf.predict( xVal_sentence)
        # get_score(np.argmax(yVal,axis=1),np.argmax(y_val_pred,axis=1))
        print(y_pred)
        y_pred = np.argmax(y_pred,axis=1)
        # get_result_file(y_pred)
        print(Counter(y_pred))
        if k==0:
            break
        # yVal = np.argmax(yVal,axis=1)
        # y_pred = clf.predict([, x_val_sentence])
        # y_val_pred = np.argmax(clf.predict([xVal_picture, xVal_sentence]),axis=1)
    #     y_val_pred = y_val_pred.reshape(-1,1)
    #     second_level_train_set[index1:index2] = y_val_pred
    #     test_nfolds_sets.append(clf.predict([x_test_picture,x_test_sentence]))
    #
    # for item in test_nfolds_sets:
    #     test_result += item
    # test_result = test_result / NUM_FOLDS
    # test_result = np.argmax(test_result,axis=1)
    # test_result = test_result.reshape(-1,1)
    # get_result_file(test_result)
    # print(test_result.shape)
    # print(second_level_train_set.shape)
    # return second_level_train_set,test_result
    return  y_pred,np.argmax(y_val_pred,axis=1),np.argmax(yVal,axis=1)
#获得分数　

def get_score(y_true,y_pred):
    if type(y_pred).__name__!="ndarray":
        y_pred=np.array(y_pred)
    if type(y_true).__name__!="ndarray":
        y_true=np.array(y_true)
    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))
    report_data= metrics.classification_report(y_true,y_pred,output_dict=True)
    df = pd.DataFrame(report_data)
    f1_score = metrics.f1_score(y_true,y_pred,average="macro")
    print("micro f1_score is {0}".format(f1_score))
    df.to_csv('./report/classification_report{0}.csv'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


def main():
    config = Config()
    token_type = "bert"
    model_names = ["gruModel","Attention_Model","Lstm_Model"]
    # model_names =["Attention_Model"]
    train_sets = []
    dev_sets = []
    dev_pred_sets = []
    test_sets = []
    config.token_type = token_type
    x_train_picture, x_train_sentence, y_train, x_test_picture, x_test_sentence = get_data(config)
    for model_name in model_names:
            config.model_name = model_name
            config.init_input()
            # train_set, test_set = get_stacking(config, x_train_picture, x_train_sentence, y_train, x_test_picture,
            #                       x_test_sentence)
            # train_sets.append(train_set)
            # test_sets.append(test_set)
            y_pred,y_val_pred,y_val  = get_stacking(config, x_train_picture, x_train_sentence, y_train, x_test_picture,x_test_sentence)

            test_sets.append(y_pred)
            dev_sets.append(y_val)
            dev_pred_sets.append(y_val_pred)
    dev_sets =np.array(dev_sets)
    dev_pred_sets = np.array(dev_pred_sets)
    test_sets = np.array(test_sets)
    # meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
    meta_dev = np.concatenate([dev_result_set.reshape(-1, 1) for dev_result_set in dev_sets], axis=1)
    # meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)
    meta_dev_pred = np.concatenate([dev_pred_set.reshape(-1, 1) for dev_pred_set in dev_pred_sets], axis=1)

    dev_label = []
    for index in meta_dev:
        frequency_dict = np.bincount(index)
        label = np.argmax(frequency_dict)
        dev_label.append(label)
    dev_label = np.array(dev_label)

    dev_pred = []
    for index in meta_dev_pred:
        frequency_dict = np.bincount(index)
        label = np.argmax(frequency_dict)
        dev_pred.append(label)
    dev_pred = np.array(dev_pred)
    get_score(dev_label,dev_pred)


    # print(meta_test.shape)
    # path = './pickle/meta_test_result.pickle'
    # pickle.dump([meta_test], open(path, 'wb'))
    # label_list = []
    # for index in meta_test:
    #     frequency_dict = np.bincount(index)
    #     label = np.argmax(frequency_dict)
    #     label_list.append(label)
    # label_list = np.array(label_list)
    # get_result_file(label_list)

    # print(meta_train.shape)
    # print(meta_test.shape)
    # path = './pickle/stacking_new_gruModel.pickle'
    # # pickle.dump([meta_train, meta_dev, meta_test, labels], open(path, 'wb'))
    # pickle.dump([meta_train, meta_test], open(path, 'wb'))
    # svc = SVC(kernel='sigmoid', gamma=1.3, C=3)
    # svc.fit(meta_train, np.array(y_train.argmax(axis=1)))
    # predictions = svc.predict(meta_test)
    # get_result_file(predictions)



if __name__=='__main__':
    main()






