
# coding: utf-8

# In[1]:


from torch.utils.data import *
import json
import torchvision.transforms as transforms
from augmentation import HorizontalFlip
from PIL import Image
NB_CLASS=61

def default_loader(path):
    return Image.open(path).convert('RGB')
    
class MyDataSet(Dataset):
    def __init__(self,json_Description,transform=None,target_transform=None,loader=default_loader,path_pre=None):
        description=open(json_Description,'r')
        imgs=json.load(description)
        image_path=[element['image_id'] for element in imgs]
        image_label=[element['disease_class'] for element in imgs]
        imgs_Norm=list(zip(image_path,image_label))
        self.imgs=imgs_Norm
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader
        self.path_pre=path_pre
    def __getitem__(self,index):
        path,label=self.imgs[index]
        img=self.loader(self.path_pre+path)
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            label=self.target_transform(label)
        return img,label
    def __len__(self):
        return len(self.imgs)
    
normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

normalize_dataset=transforms.Normalize(
    mean=[0.463,0.400, 0.486],
    std= [0.191,0.212, 0.170]
)

def preprocesswithoutNorm(image_size):
    return transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor() 
    ])

def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_with_augmentation_withoutNorm( image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor()
    ])

