

#cinfig_文件
# dense_参数
class Config(object):
    def __init__(self):
        # input configuration
        #dense model
        self.concat_1 =None
        self.rnn_dro_1 =None
        self.rnn_dro_2 = None
        self.concat_dropout_1 = None
        self.concat_dropout_2 = None
        self.concat_dropout_3 = None
        self.ker_reg_1 = float
        self.ker_reg_2 = float
        self.ker_reg_3 = float
        self.model_name =None
        self.only_sentence = False
        self.token_number =None
        self.token_feature_vector =None
        self.token_type = None
        self.filepath = "./Model/{0}.hdf5".format(self.model_name)

        self.Binary_hidden1_1 = None
        self.Binary_hidden1_2 = None
        self.Binary_dro_1_1 = None
        self.Binary_dro_1_2 = None
        self.Binary_act_1_1 = "tanh"
        self.Binary_act_1_2 = "tanh"
    def init_input(self):
        if  self.token_type =="bert":
            print("+" * 50)
            self.token_number = 128
            self.token_feature_vector = 768
            if self.model_name=="Attention_Model":
                self.concat_1 = 128
                self.rnn_dro_1 = 0.3
                self.rnn_dro_2 = 0.3
                self.concat_dropout_1 = 0.4
                self.concat_dropout_2 = 0.3
                self.concat_dropout_3 = 0.1
                self.ker_reg_1 = 1e-2
                self.ker_reg_2 = 1e-2
                self.ker_reg_3 = 1e-3
            if self.model_name =="gruModel":
                self.concat_1 = 128
                self.rnn_dro_1 = 0.2
                self.rnn_dro_2 = 0.3
                self.concat_dropout_1 = 0.3
                self.concat_dropout_2 = 0.2
                self.concat_dropout_3 = 0.0
                self.ker_reg_1 = 1e-2
                self.ker_reg_2 = 1e-3
                self.ker_reg_3 = 1e-4
            if self.model_name =="Lstm_Model":
                self.concat_1 = 128
                self.rnn_dro_1 = 0.2
                self.rnn_dro_2 = 0.2
                self.concat_dropout_1 = 0.3
                self.concat_dropout_2 = 0.2
                self.concat_dropout_3 = 0.0
                self.ker_reg_1 = 1e-3
                self.ker_reg_2 = 1e-3
                self.ker_reg_3 = 1e-4

        if  self.token_type =="elmo":
            print("-"*50)
            self.token_number = 96
            self.token_feature_vector = 1024
            if self.model_name == "Attention_Model":
                self.concat_1 = 128
                self.rnn_dro_1 = 0.3
                self.rnn_dro_2 = 0.3
                self.concat_dropout_1 = 0.4
                self.concat_dropout_2 = 0.3
                self.concat_dropout_3 = 0.1
                self.ker_reg_1 = 1e-2
                self.ker_reg_2 = 1e-2
                self.ker_reg_3 = 1e-3
            if self.model_name == "gruModel":
                self.concat_1 = 128
                self.rnn_dro_1 = 0.3
                self.rnn_dro_2 = 0.3
                self.concat_dropout_1 = 0.3
                self.concat_dropout_2 = 0.2
                self.concat_dropout_3 = 0.0
                self.ker_reg_1 = 1e-2
                self.ker_reg_2 = 1e-3
                self.ker_reg_3 = 1e-3
            if self.model_name == "Lstm_Model":
                self.concat_1 = 128
                self.rnn_dro_1 = 0.2
                self.rnn_dro_2 = 0.2
                self.concat_dropout_1 = 0.3
                self.concat_dropout_2 = 0.2
                self.concat_dropout_3 = 0.0
                self.ker_reg_1 = 1e-3
                self.ker_reg_2 = 1e-3
                self.ker_reg_3 = 1e-4
        if self.model_name=="Dense_Model":
            self.monitor = "val_acc"
            self.Binary_hidden1_1 = 256
            self.Binary_hidden1_2 = 32
            self.Binary_dro_1_1 = 0.4
            self.Binary_dro_1_2 = 0.2
            self.Binary_act_1_1 = "relu"
            self.Binary_act_1_2 = "tanh"