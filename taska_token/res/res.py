import numpy as np 
import pandas as pd 
import time
from scipy.stats import mode
from collections import Counter
def  get_result_file(y_pred):
    pd_reslut = pd.DataFrame()
    for index in y_pred:
		
        every_result = "{0}_9999_9999".format(index)
        pd_reslut = pd_reslut.append(pd.Series(every_result),ignore_index=True)
    pd_reslut.to_csv("./answer{0}.txt".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),index=None,header=0)

meta_test = pd.read_csv("./answer.csv",header=None)
meta_test =np.array(meta_test).tolist()
#label_list = mode(meta_test.transpose())[0][0]


label_list = []
for index in meta_test:
	label = mode(index)[0][0]
	print(label)
	label_list.append(label)
label_list = np.array(label_list)
print(Counter(label_list))
get_result_file(label_list)
