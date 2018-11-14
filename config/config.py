
# coding: utf-8

# In[ ]:


import datetime
class BaseConfig():
    def __init__(self,modelName):
        #模型名称
        self.modelName=modelName
        
        #各个存储路径名称
        self.img_train_pre='../data/AgriculturalDisease_trainingset/images/'
        self.img_val_pre='../data/AgriculturalDisease_validationset/images/'
        self.img_test_pre='../data/AgriculturalDisease_testA/images/'
        self.annotation_train='../data/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
        self.annotation_val='../data/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'

        #当前时间
        self.date=str(datetime.date.today())
        
        self.tensorBoard_path='../log/'+self.date+'/'+self.modelName+'/'+'tensorBoardX/'
        self.txtLog_path='../log/'+self.date+'/'+self.modelName+'/'+'txtLog/'+self.date+'_train.txt'
        self.best_path_pre='../model/'+self.date+'/'+self.modelName+'/'
        self.best_acc_path='../model/'+self.date+'/'+self.modelName+'/'+self.date+'_acc.pth'
        self.best_loss_path='../model/'+self.date+'/'+self.modelName+'/'+self.date+'_loss.pth'
        self.submit_path='../submit/'+self.date+'/result.json'
        self.SEED=666
        self.img_size=224
        self.batch_size=64
        self.num_class=61
        
class ResNet50Config(BaseConfig):
    def __init__(self):
        super().__init__('Resnet50')
        self.img_size=224
        
class ResNet101Config(BaseConfig):
    def __init__(self):
        super().__init__('Resnet101')
        self.img_size=224

class ResNet150Config(BaseConfig):
    def __init__(self):
        super().__init__('Resnet150')
        self.img_size=224
        
class DenseNet121Config(BaseConfig):
    def __init__(self):
        super().__init__('Densenet121')
        self.img_size=224
        
class DenseNet161Config(BaseConfig):
    def __init__(self):
        super().__init__('Densenet161')
        self.img_size=224
        
class DenseNet201Config(BaseConfig):
    def __init__(self):
        super().__init__('Densenet201')
        self.img_size=224    

class InceptionV3Config(BaseConfig):
    def __init__(self):
        super().__init__('InceptionV3')
        self.img_size=299
        self.batch_size=32
        
class InceptionResnetV2Config(BaseConfig):
    def __init__(self):
        super().__init__('InceptionResnetv2')
        self.img_size=299
        self.batch_size=32

resnet50Config=ResNet50Config()
resnet101Config=ResNet101Config()
resnet150Config=ResNet150Config()
densenet121Config=DenseNet121Config()
densenet161Config=DenseNet161Config()
densenet201Config=DenseNet201Config()
inceptionv3Config=InceptionV3Config()
inceptionResnetv2Config=InceptionResnetV2Config()
        

