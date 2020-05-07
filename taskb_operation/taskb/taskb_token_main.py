from __future__ import print_function
import h5py
import  numpy as np
from Taskb_My_Config import Config
from Attention_layer import AttentionM, cnnModel
import  pandas as pd
import  matplotlib
matplotlib.use('Agg')
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
import  re
from token_mylayer import Attention_model,Lstm_Model,Sentence_BinaryRelevance_Model,binary_focal_loss,gruModel,Picture_BinaryRelevance_Model
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
def get_picture_matrix(task_all_train,config):
    sentence_number=0
    train_picture=[]
    y_train = []
    for index, row in task_all_train.iterrows():   #获取Humour，Sarcasm，offensive标签
        image_name = row[0]
        Humour_label = int(row[1])
        Sarcasm_label = int(row[2])
        offensive_label = int(row[3])
        #不属于
        # other_label =  1
        if Humour_label!=0:
            Humour_label=1
        if Sarcasm_label!=0:
            Sarcasm_label=1
        if offensive_label!=0:
            offensive_label =1
        img_path = source_path + image_name
        image = keras.preprocessing.image.load_img(img_path,target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image= image.astype('float32')
        pic_vector = image/ 255.  # 去均值中心化，preprocess_input函数详细功能见注释
        train_picture.append(pic_vector)
        y_train.append([Humour_label,Sarcasm_label,offensive_label])
        sentence_number+=1
    train_picture = np.vstack(train_picture)
    y_train = np.array(y_train)
    print(Counter(y_train[:,0]))
    print(Counter(y_train[:,1]))
    print(Counter(y_train[:,2]))
    return  train_picture,y_train

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

#获得文件
def  get_result_file(y_pred):
    pd_reslut = pd.DataFrame()
    pd_report_reslut =pd.DataFrame()
    for index in y_pred:
        Humour_label = index[0]
        Sarcasm_label = index[1]
        offensive_label = index[2]
        motivational_label = index[3]
        every_result = "9_{0}{1}{2}{3}_9999".format(Humour_label,Sarcasm_label,offensive_label,motivational_label)
        every_result_report = [Humour_label,Sarcasm_label,offensive_label,motivational_label]
        pd_reslut = pd_reslut.append(pd.Series(every_result),ignore_index=True)
        pd_report_reslut = pd_reslut.append(pd.Series(every_result_report),ignore_index=True)
    pd_reslut.to_csv("./report/answer{0}.txt".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),index=None,header=0)
    pd_report_reslut.to_csv(("./report/reslut_org.csv"))
    print(Counter(y_pred[:,0]))
    print(Counter(y_pred[:,1]))
    print(Counter(y_pred[:,2]))
    print(Counter(y_pred[:,3]))
#将非0的label转换为1
def conver_label(label_list):
    if type(label_list).__name__!="list":
        label_list = label_list.tolist()
    for index in label_list:
        index[0] = 1 if index[0]!=0 else 0
        index[1] = 1 if index[1]!=0 else 0
        index[2] = 1 if index[2] != 0 else 0
        index[3] = 1 if index[3] != 0 else 0
    label_array=np.array(label_list)
    return  label_array
#
#将预测转换为对应的label 标签
def get_y_pred_label(label_list):
    if type(label_list).__name__!="list":
        label_list = label_list.tolist()
    for index in label_list:
        index[0] = 1 if index[0]>=0.5 else 0
        index[1] = 1 if index[1]>=0.5 else 0
        index[2] = 1 if index[2]>=0.5 else 0
        index[3] = 1 if index[3]>=0.5 else 0
    label_array = np.array(label_list)
    return  label_array
#获得multilabel
def get_multilabel(train_label):
    label_list =[]
    for index,row in train_label.iterrows():
        Humour_label= row["Humour"]
        Sarcasm_label=row["Sarcasm"]
        offensive_label= row["offensive"]
        motivational_label=row["motivational"]
        label_list.append([Humour_label,Sarcasm_label,offensive_label,motivational_label])
    label_array = conver_label(label_list)
    return  label_array
def get_data(config):
    # 训练集的两个分量
    #如果是二维关系就用sentence_embedding and picture_vertor
    if (config.model_name=="Sentence_BinaryRelevance_Model")|(config.model_name=="Picture_BinaryRelevance_Model"):

       # 获得句向量和图片向量
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


    else:   #其他的都获得图片和token_embedding
    #获得句向量和图片向量
        if config.token_type == "elmo":
            trian_sentence_path = "../../data/train_data/sentence_data/elmo_token96_train.pickle"
            x_train_sentence = pickle.load(open(trian_sentence_path, 'rb'))
            test_sentence_path = "../../data/test_data/sentence_data/elmo_token96_test.pickle"
            x_test_sentence = pickle.load(open(test_sentence_path, 'rb'))
            x_train_picture = pd.read_csv("../../data/train_data/sentence_data/data_have_sentence.csv", header=0,
                                          delimiter=",")
            x_train_picture = get_picture_numpy(x_train_picture)
            print(x_train_picture.shape)
            x_test_picture = pd.read_csv("../../data/test_data/sentence_data/test_data_all.csv", delimiter=",",
                                         header=0)
            x_test_picture = get_picture_numpy(x_test_picture)
        else:
            trian_sentence_path = "../../data/train_data/sentence_data/bert_token128_train_new.pickle"
            x_train_sentence = pickle.load(open(trian_sentence_path, 'rb'))

            x_train_picture = pd.read_csv("../../data/train_data/sentence_data/data_have_sentence.csv", header=0, delimiter=",")
            x_train_picture = get_picture_numpy(x_train_picture)
            print(x_train_picture.shape)

            test_sentence_path = "../../data/test_data/sentence_data/bert_token128_test.pickle"
            x_test_sentence = pickle.load(open(test_sentence_path, 'rb'))

            x_test_picture = pd.read_csv("../../data/test_data/sentence_data/test_data_all.csv", delimiter=",", header=0)
            x_test_picture = get_picture_numpy(x_test_picture)


    #获得train_label val_label
    train_label = pd.read_csv("../../data/train_data/sentence_data/data_have_sentence.csv",header=0,delimiter=",")
    y_train =get_multilabel(train_label)
    x_train_sentence = np.array(x_train_sentence)
    x_test_sentence = np.array(x_test_sentence)
    print(x_train_sentence.shape)
    print(x_train_picture.shape)
    print(y_train.shape)
    print(x_test_sentence.shape)

    return  x_train_picture,x_train_sentence,y_train,x_test_picture,x_test_sentence


def panda_str_to_list_matrix(rev):
    new_rev = []
    for i in rev:
        new_i = []
        # 删除掉符号[]
        i = re.sub('[\[\]]', "", i)
        # 获得只有数字的list
        i = i.split(",")
        # 将数字由str格式转换成float格式
        i = [float(num) for num in i]
        print(i)
        new_rev.append(i)
    return new_rev


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
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation acc values
    plt.subplot(2, 2, 2)
    plt.plot(history.history['Humour_acc'])
    plt.plot(history.history['val_Humour_acc'])
    plt.title('Model Acc Humour')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation acc values
    plt.subplot(2, 2, 3)
    plt.plot(history.history['Sarcasm_acc'])
    plt.plot(history.history['val_Sarcasm_acc'])
    plt.title('Model Acc Sarcasm')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation acc values
    plt.subplot(2, 2, 4)
    plt.plot(history.history['Offensive_acc'])
    plt.plot(history.history['val_Offensive_acc'])
    plt.title('Model Acc Offensive')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./picture/picture+{0}.jpg'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    plt.close()

#预留test空间
def get_stacking(config,x_train_picture,x_train_sentence,y_train, x_test_picture,x_test_sentence, n_folds=5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train_picture.shape[0], x_test_picture.shape[0],


    second_level_train_set = np.zeros((train_num, 3))
    test_result = np.zeros((test_num, 3))
    test_nfolds_sets = []
    dev_nfolds_stes = []
    for k in range(NUM_FOLDS):
        validationSize = int(len(x_train_sentence)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k + 1)

        xTrain_sentence= np.vstack((x_train_sentence[:index1],x_train_sentence[index2:]))
        xTrain_picture = np.vstack((x_train_picture[:index1], x_train_picture[index2:]))
        yTrain = np.vstack((y_train[:index1], y_train[index2:]))
        xVal_sentence = x_train_sentence[index1:index2]
        xVal_picture = x_train_picture[index1:index2]
        yVal = y_train[index1:index2]

        Train_Humour_label = np.array(yTrain[:, 0])
        Train_Sarcasm_label = np.array(yTrain[:, 1])
        Train_offensive_label = np.array(yTrain[:, 2])
        Train_motivational_label = np.array(yTrain[:, 3])

        Val_Humour_label = np.array(yVal[:, 0])
        Val_Sarcasm_label = np.array(yVal[:, 1])
        Val_offensive_label = np.array(yVal[:, 2])
        Val_motivational_label = np.array(yVal[:, 3])
        print(xTrain_picture.shape)
        print(xTrain_sentence.shape)
        if config.model_name=="Sentence_BinaryRelevance_Model":
            test_pred_0, test_pred_1, test_pred_2, test_pred_3,val_pred_0, val_pred_1, val_pred_2,val_pred_3\
                = Sentence_BinaryRelevance_Model(k,config,xTrain_picture,xTrain_sentence,yTrain,xVal_picture, xVal_sentence,yVal,x_test_picture,x_test_sentence)
        elif config.model_name=="Picture_BinaryRelevance_Model":
            test_pred_0, test_pred_1, test_pred_2, test_pred_3,val_pred_0, val_pred_1, val_pred_2,val_pred_3\
                = Picture_BinaryRelevance_Model(k,config,xTrain_picture,xTrain_sentence,yTrain,xVal_picture, xVal_sentence,yVal,x_test_picture,x_test_sentence)

        else:

            if config.model_name == "Attention_model":
                clf = Attention_model(config)
            if config.model_name =="Lstm_Model":
                clf = Lstm_Model(config)
            if config.model_name=="gruModel":
                clf = gruModel(config)
            config.filepath = "./Model/{0}+n_fold+{1}.hdf5".format(k, config.model_name)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, mode='auto')
            checkpoint = ModelCheckpoint(config.filepath, monitor="val_loss", verbose=1, save_best_only=True,
                                         mode='auto')
            early_point = EarlyStopping(monitor="val_loss", mode='auto', patience=10)
            callbacks_list = [checkpoint, early_point,reduce_lr]

            history = clf.fit([xTrain_picture, xTrain_sentence],
                              [Train_Humour_label,Train_Sarcasm_label,Train_offensive_label,Train_motivational_label],
                              batch_size=100, epochs=50, validation_data=([xVal_picture,xVal_sentence],
                                                                           [Val_Humour_label,Val_Sarcasm_label,Val_offensive_label,Val_motivational_label]),
                              callbacks=callbacks_list, verbose=1)
            get_acc_loss_img(history)
            if config.model_name == "Attention_model":
                clf = load_model(config.filepath,custom_objects={'AttentionM':AttentionM,'binary_focal_loss_fixed': binary_focal_loss()})
                # clf = load_model(config.filepath,custom_objects={'AttentionM':AttentionM,})
            else:
                # clf = load_model(config.filepath, custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
                clf = load_model(config.filepath)
            val_pred_0, val_pred_1, val_pred_2,val_pred_3 = clf.predict([xVal_picture,xVal_sentence])
            test_pred_0,test_pred_1,test_pred_2,test_pred_3 = clf.predict(([x_test_picture,x_test_sentence]))
            # y_pred = np.array(y_pred)
        #     get_acc_loss_img(history)
        y_pred = np.concatenate((val_pred_0,val_pred_1),axis=1)
        y_pred = np.concatenate((y_pred, val_pred_2),axis=1)
        y_pred = np.concatenate((y_pred, val_pred_3), axis=1)

        y_test_pred = np.concatenate((test_pred_0, test_pred_1), axis=1)
        y_test_pred = np.concatenate((y_test_pred, test_pred_2), axis=1)
        y_test_pred = np.concatenate((y_test_pred, test_pred_3), axis=1)
        y_test_pred = get_y_pred_label(y_test_pred)
        get_result_file(y_test_pred)

        print(y_pred.shape)
        # print(y_pred)
        if type(y_pred).__name__ !="ndarray":
            y_pred = y_pred.toarray()

        y_val = np.array(yVal)
        y_pred = get_y_pred_label(y_pred)
        print(y_pred)

            #对每一个标签求f1 recall hamloss
        avg_f1=0
        avg_recall=0
        ave_Hamming_loss=0
        for  i in range(4):
            f1,recall,Hamming_loss=get_score(i,y_val[:,i],y_pred[:,i])
            avg_f1+=f1
            ave_Hamming_loss+=Hamming_loss
            avg_recall+=recall
        avg_f1=avg_f1/4
        avg_recall=avg_recall/4
        ave_Hamming_loss=ave_Hamming_loss/4
        print("avg_f1 is {}".format(avg_f1))
        print("avg_recall is {}".format(avg_recall))
        print("ave_Hamming_loss is {}".format(ave_Hamming_loss))
        df = pd.DataFrame()
        df = df.append(pd.Series({"avg_f1":[avg_f1],"avg_recall":[avg_recall],"Hamming_loss":[ave_Hamming_loss]}),ignore_index=True)

        df.to_csv('./report/classification_report{0}.csv'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        exit()

        # second_level_train_set[index1:index2] = clf.predict(xVal)
        # dev_nfolds_stes.append(clf.predict(x_val))
        #test_nfolds_sets[:, i] = clf.predict(x_test)
    # for item in test_nfolds_sets:
    #     test_result += item
    # test_result = test_result / NUM_FOLDS

    # for item in dev_nfolds_stes:
    #     dev_result += item
    # dev_result = dev_result / NUM_FOLDS
    # return second_level_train_set, dev_result,test_result
#获得分数　

def get_score(i,y_true,y_pred):
    if type(y_pred).__name__!="ndarray":
        y_pred=np.array(y_pred)
    if type(y_true).__name__!="ndarray":
        y_true=np.array(y_true)
    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))
    report_data = metrics.classification_report(y_true,y_pred,output_dict=True)
    df = pd.DataFrame(report_data)
    Hamming_loss = metrics.hamming_loss(y_true, y_pred)
    f1_score = metrics.f1_score(y_true,y_pred,average="macro")
    recall_score = metrics.recall_score(y_true, y_pred, average="macro")
    # df = df.append(pd.Series({"Hamming_loss":[Hamming_loss]}),ignore_index=True)

    df.to_csv('./report/classification_report+{0}+{1}.csv'.format(i,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    return  f1_score,recall_score,Hamming_loss

def operation_function(config,x_train_picture, x_train_sentence, y_train, x_val_picture,x_val_sentence, y_val):
    config.init_input()
    get_stacking(config,x_train_picture,x_train_sentence,y_train,x_val_picture,x_val_sentence,y_val)



# def main():
#     config = Config()
#     sentence_types=["bert","infersent"]
#     model_names = ["Dense_model", "DecisionTree_model"]
#     train_sets = []
#     dev_sets = []
#     test_sets = []
#     only_sentence_all = [True]
#     for sentence_type in  sentence_types:
#         for model_name in model_names:
#             for only_sentence in only_sentence_all:
#                 config.only_sentence = only_sentence
#                 config.model_name = model_name
#                 config.sentence_type = sentence_type
#                 x_sentence, x_picture, y_train, x_val_sentence, x_val_picture, y_val = get_data(config)
#                 y_val = np.array(y_val)
#                 train_set, dev_set, test_set = operation_function(config, x_sentence, x_picture, y_train, x_val_picture,
#                                                                   x_val_sentence, y_true)
#                 train_sets.append(train_set)
#                 dev_sets.append(dev_set)
#                 test_sets.append(test_set)
#     meta_train = np.concatenate([result_set.reshape(-1, 4) for result_set in train_sets], axis=1)
#     meta_dev = np.concatenate([dev_result_set.reshape(-1, 4) for dev_result_set in dev_sets], axis=1)
#     #meta_test = np.concatenate([y_test_set.reshape(-1, 4) for y_test_set in test_sets], axis=1)
#     path = './pickle/stacking_new_elmo.pickle'
#     # pickle.dump([meta_train, meta_dev, meta_test, labels], open(path, 'wb'))
#     pickle.dump([meta_train, meta_dev, ], open(path, 'wb'))
#     svc = SVC(kernel='sigmoid', gamma=1.3, C=3)
#     svc.fit(meta_train, np.array(labels.argmax(axis=1)))
#     predictions = svc.predict(meta_dev)

if __name__=='__main__':
    config = Config()
    token_type = "elmo"
    model_names = ["Attention_model","Sentence_BinaryRelevance_Model","Lstm_Model","gruModel"]
    model_names = ["gruModel"]
    for model_name in model_names:
        config.model_name = model_name
        config.token_type = token_type
        x_train_picture, x_train_sentence, y_train, x_test_picture,x_test_sentence = get_data(config)
        config.init_input()
        get_stacking(config, x_train_picture, x_train_sentence, y_train, x_test_picture,x_test_sentence)


