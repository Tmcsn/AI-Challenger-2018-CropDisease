
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import datetime
import os
from PIL import Image
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import tqdm
import cv2
from torchsummary import summary
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('../config/')
import config 


# In[2]:


'''
训练过程中保存loss和acc
'''
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return float(self.total_value)/ self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)
    
'''
保存训练快照
'''
def snapshot(savepathPre,savePath,state):    
    if not os.path.exists(savepathPre):
        os.makedirs(filePath)
    torch.save(state, savePath)

    
'''
保存自定义log信息 
'''
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    


'''
预测data在model上的结果
输出两个GPU上的数组 所有的label值和所有的预测值
'''
def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            all_labels.append(labels)
            inputs = Variable(inputs).cuda()
            outputs = model(inputs)
            all_outputs.append(outputs.data.cpu())
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()
    return all_labels, all_outputs 

'''
将 a 扩展一维  并和acc在最后一维上做连接操作 
'''
def safe_stack_2array(acc, a):
    a = a.unsqueeze(-1) # 在最后一维扩充
    if acc is None:
        return a
    return torch.cat((acc, a), dim=acc.dim() - 1)

'''
TTA时使用不同的augmentation方法生成不同的dataLoader 并将预测结果连接
'''
def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        print('predict finish')
        prediction = safe_stack_2array(prediction, px)
    return lx, prediction



'''
prob:对各个类别预测的概率，这里的prob是经过softmax之后的结果  


'''
def calibrate_probs(train_df,val_df,prob,NB_CLASS):
    calibrated_prob = np.zeros_like(prob)
    nb_train = train_df.shape[0]
    for class_ in range(NB_CLASS): # enumerate all classes 这里有61类 其他
        prior_y0_train = (train_df['disease_class'] == class_).mean() #类别为class_的先验
        prior_y1_train = 1 - prior_y0_train
        prior_y0_test=(val_df['disease_class'] ==class_).mean()
        prior_y1_test=1-prior_y0_test
        for i in range(prob.shape[0]): # enumerate every probability for a class
            predicted_prob_y0 = prob[i, class_]
            calibrated_prob_y0 = calibrate(
                prior_y0_train, prior_y0_test,
                prior_y1_train, prior_y1_test,                
                predicted_prob_y0)
            calibrated_prob[i, class_] = calibrated_prob_y0
    return calibrated_prob

def calibrate(prior_y0_train, prior_y0_test,
              prior_y1_train, prior_y1_test,
              predicted_prob_y0):
    predicted_prob_y1 = (1 - predicted_prob_y0)
    
    p_y0 = prior_y0_test * (predicted_prob_y0 / prior_y0_train)
    p_y1 = prior_y1_test * (predicted_prob_y1 / prior_y1_train)
    return p_y0 / (p_y0 + p_y1)  # normalization


'''
将44,45类删除

'''

def deleteNosiseType():    
    description_train=open('../data/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json','r')
    description_val=open('../data/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json','r')
    img_train=json.load(description_train)
    img_val=json.load(description_val)
    for element in img_train:
        if element['disease_class']==45:
            img_train.remove(element)
            continue
        if element['disease_class']==44:
            img_train.remove(element)
            continue 
        if element['disease_class']>45:
            element['disease_class'] = element['disease_class']-2
    with open("../data/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations_deleteNoise.json",'w') as f:
        json.dump(img_train,f,ensure_ascii=False)
    with open("../data/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations_deleteNoise.json",'w') as f:
        json.dump(img_val,f,ensure_ascii=False)
 

'''
分割验证集
'''

def validateDataSplit(dataFrame,num=3):
    split_fold = StratifiedKFold(n_splits=3)
    folds_indexes = split_fold.split(X=dataFrame["image_id"],y=dataFrame["disease_class"])
    folds_indexes = np.array(list(folds_indexes))
    trainIndex=folds_indexes[0][0]
    valiIndex=folds_indexes[0][1]
    trainDataFrame=dataFrame.iloc[trainIndex]
    valDataFrame=dataFrame.iloc[valiIndex]
    train=trainDataFrame.to_json("../data/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations_train.json",orient='records',force_ascii=False)
    val=valDataFrame.to_json("../data/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations_val.json",orient='records',force_ascii=False)

'''
get std and mean  未测试正确性
可以仔细看一下 tqdm的使用方法

'''
def getstdAndMean(imgPath,dataFrame):
    CROP_SIZE=224
    data=[]
    for fileName in tqdm.tqdm_notebook(dataFrame['image_id'],miniters=256):
        img=cv2.imread(imgPath+fileName)
        data.append(cv2.resize(img,(CROP_SIZE,CROP_SIZE)))
    data = np.array(data, np.float32) / 255
    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

    
'''
获得结果的混淆矩阵
'''    

def processCMatrix(y_true,y_predict):
    matrix=confusion_matrix(y_true, y_predict)
    result={}
    length=matrix.shape[0]
    for i in range(length):
        result.setdefault('precision',[]).append(matrix[i][i]/sum(matrix[i]))
        result.setdefault('recall',[]).append(matrix[i][i]/sum(matrix[:,i]))
    return matrix,result


'''
label smooth
'''
def labelShuffling(dataFrame,outputPaht="../data/AgriculturalDisease_trainingset/",outputName="AgriculturalDisease_train_Shuffling_annotations.json",groupByName='disease_class'):
    groupDataFrame=dataFrame.groupby(by=[groupByName])
    labels=groupDataFrame.size()
    print("length of label is ",len(labels))
    maxNum=max(labels)
    lst=pd.DataFrame(columns=["disease_class","image_id"])
    for i in range(len(labels)):
        print("Processing label  :",i)
        tmpGroupBy=groupDataFrame.get_group(i)
        createdShuffleLabels=np.random.permutation(np.array(range(maxNum)))%labels[i]
        print("Num of the label is : ",labels[i])
        lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels],ignore_index=True)
        print("Done")
    lst.to_json(outputPaht+outputName,orient="records",force_ascii=False)

