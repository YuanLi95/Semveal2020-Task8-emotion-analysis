import  numpy as np
my_list = []
a = [1,1,2,4,5,6,]
b = [0,1,2,4,0,6,]
c = [1,0,0,0,0,0]
my_list =[a,b,c]
my_list = np.array(my_list)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in my_list], axis=1)
label_list = []
for index in meta_test:
    frequency_dict=np.bincount(index)
    label = np.argmax(frequency_dict)
    label_list.append(label)
print(label_list)