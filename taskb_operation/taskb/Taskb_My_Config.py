

#cinfig_文件
# dense_参数
class Config(object):
    def __init__(self):
        # input configuration
        #BinaryRelevance_Model
        self.monitor = None
        self.token_number = 128
        self.token_feature_vector = 768
        self.Binary_hidden1_1 =None
        self.Binary_hidden1_2 = None
        self.Binary_dro_1_1 = None
        self.Binary_dro_1_2 = None
        self.Binary_act_1_1 ="tanh"
        self.Binary_act_1_2 = "tanh"

        self.Binary_hidden2_1 = None
        self.Binary_hidden2_2 = None
        self.Binary_dro_2_1 = None
        self.Binary_dro_2_2 = None
        self.Binary_act_2_1 = "tanh"
        self.Binary_act_2_2 = "tanh"

        self.Binary_hidden3_1 = None
        self.Binary_hidden3_2 = None
        self.Binary_dro_3_1 = None
        self.Binary_dro_3_2 = None
        self.Binary_act_3_1 = "tanh"
        self.Binary_act_3_2 = "tanh"
        self.model_name =None
        self.token_type  ="bert"
        self.filepath = "./Model/{0}.hdf5".format(self.model_name)

    def init_input(self):
        if self.token_type =="bert":
            self.token_number = 128
            self.token_feature_vector = 768
        elif  self.token_type =="elmo":
            self.token_number = 96
            self.token_feature_vector = 1024
        if self.model_name=="Sentence_BinaryRelevance_Model":
            print("+" * 50)
            #dense  模型参数
            self.dense_hidden_1 = 256
            self.dense_hidden_2 = 64
            self.act  = "relu"
            self.drop_out1 = 0.0
            self.drop_out2 = 0.3
            self.drop_out3 = 0.2

            # BinaryRelevance_Model
            self.monitor = "val_acc"
            self.Binary_hidden1_1 = 256
            self.Binary_hidden1_2 = 32
            self.Binary_dro_1_1 = 0.5
            self.Binary_dro_1_2 = 0.2
            self.Binary_act_1_1 = "relu"
            self.Binary_act_1_2 = "tanh"

            self.Binary_hidden2_1 = 256
            self.Binary_hidden2_2 = 32
            self.Binary_dro_2_1 = 0.5
            self.Binary_dro_2_2 = 0.3
            self.Binary_act_2_1 = "relu"
            self.Binary_act_2_2 = "tanh"

            self.Binary_hidden3_1 = 128
            self.Binary_hidden3_2 = 32
            self.Binary_dro_3_1 = 0.3
            self.Binary_dro_3_2 = 0.3
            self.Binary_act_3_1 = "relu"
            self.Binary_act_3_2 = "relu"

            self.Binary_hidden4_1 = 128
            self.Binary_hidden4_2 = 32
            self.Binary_dro_4_1 = 0.3
            self.Binary_dro_4_2 = 0.2
            self.Binary_act_4_1 = "relu"
            self.Binary_act_4_2 = "relu"

        if self.model_name=="Picture_BinaryRelevance_Model":
            print("+" * 50)
            #dense  模型参数
            self.dense_hidden_1 = 256
            self.dense_hidden_2 = 64
            self.act  = "relu"
            self.drop_out1 = 0.0
            self.drop_out2 = 0.3
            self.drop_out3 = 0.2

            self.monitor = "val_acc"
            self.Binary_hidden1_1 = 128
            self.Binary_hidden1_2 = 32
            self.Binary_dro_1_1 = 0.4
            self.Binary_dro_1_2 = 0.2
            self.Binary_act_1_1 = "relu"
            self.Binary_act_1_2 = "relu"

            self.Binary_hidden2_1 = 128
            self.Binary_hidden2_2 = 32
            self.Binary_dro_2_1 = 0.4
            self.Binary_dro_2_2 = 0.3
            self.Binary_act_2_1 = "relu"
            self.Binary_act_2_2 = "relu"

            self.Binary_hidden3_1 = 128
            self.Binary_hidden3_2 = 32
            self.Binary_dro_3_1 = 0.3
            self.Binary_dro_3_2 = 0.2
            self.Binary_act_3_1 = "relu"
            self.Binary_act_3_2 = "relu"

            self.Binary_hidden4_1 = 128
            self.Binary_hidden4_2 = 32
            self.Binary_dro_4_1 = 0.3
            self.Binary_dro_4_2 = 0.2
            self.Binary_act_4_1 = "relu"
            self.Binary_act_4_2 = "relu"

