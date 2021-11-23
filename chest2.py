# %%

try:
  %load_ext autotime
except:
  print("Console warning-- Autotime is jupyter platform specific")


# %%
import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import comet_ml
from comet_ml import Experiment
import icecream as ic
from IPython.display import Audio, display
from torch.utils.data import DataLoader, Dataset, random_split
import torch
torch.cuda._initialized = True
import time
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from pydicom import dcmread
from matplotlib import pyplot as plt 


# %%
import pandas as pd 
df = pd.read_csv('/media/gyasis/Drive 2/Data/vinbigdata/train.csv')
df.head(10)

# %%
df = df[['image_id', 'class_name','class_id']]
torch.cuda.empty_cache() 
# %%
def build_path(x):
    path_ = '/media/gyasis/Drive 2/Data/vinbigdata/train/'
    filetype = '.dicom'
    x = (path_+x+filetype)
    return x

# %%
import os.path

# %%
df['imagepath'] = df['image_id'].apply(lambda x: build_path(x))
df = df[['imagepath','class_name','class_id']]
df.head()

# %% 
pd.get_dummies(df['class_name'])
df1 = pd.get_dummies(df['class_id'].astype(str))
# %%
#mapping for later use
disease= ["Aortic enlargement"
,"Atelectasis"
,"Calcification"
,"Cardiomegaly"
,"Consolidation"
,"ILD"
,"Infiltration"
,"Lung Opacity"
,"Nodule/Mass"
,"Other lesion"
,"Pleural effusion"
,"Pleural thickening"
,"Pneumothorax"
,"Pulmonary fibrosis"
,"No_finding"]

#map df.class_id to disease
# df['class_id_test'] = df['class_id'].map(lambda x: disease[x])
df.head()
df1.columns = df1.columns.astype(int).map(lambda x: disease[x])

s_array = np.array(df1)
# %%
def get_class_frequencies():
  positive_freq = s_array.sum(axis=0) / s_array.shape[0]
  negative_freq = np.ones(positive_freq.shape) - positive_freq
  return positive_freq, negative_freq

p,n = get_class_frequencies()
# %%
data = pd.DataFrame({"Class": df1.columns, "Label": "Positive", "Value": p})
data = data.append([{"Class": df1.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(n)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value",hue="Label", data=data)
# %%
pos_weights = n
neg_weights = p
pos_contribution = p * pos_weights
neg_contribution = n * neg_weights
print(p)
print(n)
print("Weight to be added:  ",pos_contribution)


# %%

data = pd.DataFrame({"Class": df1.columns, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": df1.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
g = sns.barplot(x="Class", y="Value",hue="Label", data=data)
# %%
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

c_transform = nn.Sequential(transforms.Resize([256,]), 
                            transforms.CenterCrop(224),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
ten = torchvision.transforms.ToTensor()

scripted_transforms = torch.jit.script(c_transform)
# %%
transform = A.Compose(
    [A.Resize(width=256,height=256, always_apply=True),
                       A.HorizontalFlip(p=0.5),
                       A.OneOf([
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.25),
                            A.RandomBrightnessContrast(p=0.1, contrast_limit=0.05, brightness_limit=0.05,),
                            A.InvertImg(p=0.02),
                       ]),
                       A.OneOf([
                           A.RandomCrop(width=224, height=224, p=0.5),
                           A.CenterCrop(width=224, height=224, p=0.5),
                           
                       ]),
                       A.Resize(width=224, height=224, always_apply=True),
                       A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                       ToTensorV2()
                    ])
# %%
class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        print('creating Dataset')
        self.df = dataset
        self.imagearray = np.asarray(dataset.imagepath)
        self.class_arr = np.asarray(dataset.class_id)
        self.transform = transform
        self.data_len = len(dataset.index)
        
    def __getitem__(self, index):
        ds = dcmread(self.imagearray[index])
        arr = ds.pixel_array
        arr = arr.astype('float')
        class_id = self.df.loc[index, 'class_id']
        arr = ten(arr)
        arr = arr.expand(3, -1,-1)
        arr = scripted_transforms(arr)
        
        return arr, class_id
        
    def __len__(self):
        return self.data_len
    
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=transform):
        self.df = dataset
        self.imagearray = np.asarray(dataset.imagepath)
        self.class_arr = np.asarray(dataset.class_id)
        self.transform = transform
        self.data_len = len(dataset.index)
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        
        ds = dcmread(self.imagearray[index])
        class_id = self.df.loc[index, 'class_id']
        arr = ds.pixel_array
        arr = np.stack((arr,)*3, axis=-1)
        arr = transform(image = arr)["image"]
        return arr, class_id

# %%
ChestData = MyDataset(df, transform=None)
#  %%
ChestData_Aug = AlbumentationsDataset(df, transform=transform)
# %%
set_batchsize = 128
# %%
from torch.utils.data import DataLoader, Dataset, random_split
num_items = len(ChestData_Aug)
num_train = round(num_items * 0.7)
num_val = num_items - num_train
train_ds, val_ds = random_split(ChestData_Aug, [num_train, num_val])
train_dataloader = DataLoader(train_ds, batch_size=set_batchsize, num_workers=4, pin_memory=True, shuffle=True)
val_dataloader = DataLoader(val_ds,batch_size=set_batchsize, num_workers=4, pin_memory=True,shuffle=False)


# %%
from torchvision import models 
import torch
model = models.resnet18(pretrained=True)

print(model)
# %%
from torch import nn as nn
num_classes = 15
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0")
model= model.to(device)

# %%
df.class_id.unique()

# %%
# torch.manual_seed(17)

experiment = Experiment(api_key="xleFjfKO3kcwc56tglgC1d3zU",
                        project_name="Chest Xray",log_code=True)

# %%
# import copy


# def visualize_augmentations(dataset, idx=12,iterate='random', samples=9, cols=3):
#     dataset = copy.deepcopy(dataset)
#     dataset.transform = transform
#     rows = samples // cols
#     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
    
    
#     for i in range(samples):
#         rando = random.randint(0,len(dataset)-1)
#         if (iterate=='random'):
#             image, _ = dataset[rando]
#             print(image.shape)
#             ax.ravel()[i].imshow(image[:,:,0],cmap='gray')
#             ax.ravel()[i].set_axis_off()
#         else:
#             image, _ = dataset[i]
#             ax.ravel()[i].imshow(image[0,:,:],cmap='gray')
#             ax.ravel()[i].set_axis_off()
            
#     plt.tight_layout()
#     filename = 'augmented_images_' + str(rando) + '.png'
#     plt.savefig(filename)
#     experiment.log_image(image_data = filename) 
#     plt.show()
    
    
# # %%
# for t in range(9):
#     visualize_augmentations(ChestData_Aug, idx=5, samples=9, cols=3)   

# # %%
# for t in range(9):
#     visualize_augmentations(ChestData, idx=5, samples=9,iterate='iterate', cols=3)
# # %%
# random.seed(42)
# visualize_augmentations(ChestData)

# %%
def training(model, train_dataloader, num_epochs):
    optimizer_name = torch.optim.SGD(model.parameters(), lr=0.01)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(pos_contribution).type(torch.FloatTensor).to(device))
    optimizer = optimizer_name
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.01,
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        
        
        for i, data, in enumerate(tqdm(train_dataloader)):
            inputs = data[0].float().to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            running_acc = correct_prediction/total_prediction
            experiment.log_metric("Train/train_accuracy", running_acc, epoch)
            try:
                experiment.log_metric("Loss/train",running_loss/i, epoch)
            except:
                print('div by zero')
            if i > 2:
                if i % 20 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))
                
        num_batches = len(train_dataloader)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        experiment.log_metric("Accuracy", acc, epoch)
        
num_epochs = 1 
training(model, train_dataloader, num_epochs)
experiment.end()
# %%
print(model)
# %%
# ----------------------------
# Inference
# ----------------------------
def inference (model, x):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in x:
      # Get the input features and target labels, and put them on thresige GPU
      inputs = data[0].float().to(device)
      labels =  data[1].to(device)
      
      
      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)
     

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
      ic.ic(prediction)
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model with the validation set

inference(model, val_dataloader)
# %%
import copy
def visualize_augmentations(dataset, idx=12, samples=9, cols=3):
    dataset = copy.deepcopy(dataset)
    dataset.transform = transform
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
    for i in range(samples):
        image, _ = dataset[i]
       
        ax.ravel()[i].imshow(image[0,:,:],cmap='gray')
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
# %%
random.seed(42)
visualize_augmentations(ChestData)
# %%
ds = dcmread(df.imagepath[5])
arr = ds.pixel_array
# arr = arr.astype('float')
arr = np.stack((arr,)*3, axis=-1)

print(arr.shape)
print(arr.dtype)

transformed = transform(image = arr)["image"]

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
visualize(transformed)