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

# setting values to rows and column variables 
# make this a function for later 


# rows = 7
# columns = 4

# path = 'data/small_collection/'
# i = 1
# for item in imagelist:
#     ds = dcmread(item)
#     # `arr` is a numpy.ndarray
#     arr = ds.pixel_array

#     # Adds a subplot at the 1st position 
#     fig.add_subplot(rows, columns, i) 
#     # showing image 
#     plt.imshow(arr, cmap="gray") 
#     plt.axis('off') 
#     i += 1

# %%
# fig2 = plt.figure(figsize=(30,60))
# arr = contents.pixel_array
# plt.imshow(arr, cmap="gray")

# %%
import pandas as pd 
df = pd.read_csv('data/train.csv')
df.head(10)

# %%
df = df[['image_id', 'class_id']]
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

# %%
df = df[['imagepath','class_id']]
# %%
df.head()

# %%
import torchvision
from torchvision import transforms

c_transform = nn.Sequential(transforms.Resize([256,]),
                                  transforms.CenterCrop(224),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
ten = torchvision.transforms.ToTensor()

scripted_transforms = torch.jit.script(c_transform)

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
        # ic.ic(arr.shape)
        arr = arr.astype('float')
        # arr = test(arr)
        # arr = torchvision.transforms.ToPILImage(arr)
        
        class_id = self.df.loc[index, 'class_id']
        # print(arr.shape)
        arr = ten(arr)
        # print(arr.shape)
        arr = arr.expand(3, -1,-1)
        # print(arr.shape)
        arr = scripted_transforms(arr)
        # print(arr.shape)
  
        return arr, class_id
        
    def __len__(self):
        return self.data_len
    
    
    
    
    
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.df = dataset
        self.file_paths = np.asarray(dataset.imagepath)
        self.labels = np.asarray(dataset.class_id)
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        labels = self.df.loc[index, 'class_id']
        file_path = self.file_paths[index]
        ds = dcmread(file_path)
        arr = ds.pixel_array
        
        return arr, labels
        
# %% 

# dir(transforms)
# %%
ChestData = MyDataset(df, transform=c_transform)
# %%
set_batchsize = 128
# %%
from torch.utils.data import DataLoader, Dataset, random_split
num_items = len(ChestData)
num_train = round(num_items * 0.7)
num_val = num_items - num_train
train_ds, val_ds = random_split(ChestData, [num_train, num_val])
train_dataloader = DataLoader(train_ds, batch_size=set_batchsize, num_workers=4, pin_memory=True, shuffle=True)
val_dataloader = DataLoader(val_ds,batch_size=set_batchsize, num_workers=4, pin_memory=True,shuffle=False)


# %%
# end_ = 5
# for i,data in enumerate(train_dataloader):
  
#     if i < end_: 
#         print(data)
#         print(type(data))
#         print(len(data))
#     else: 
#         break
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
# from torchinfo import summary
# summary(model, input_size = (set_batchsize, 3 ,224,224), device = device.type)
# %%
# torch.manual_seed(17)

experiment = Experiment(api_key="xleFjfKO3kcwc56tglgC1d3zU",
                        project_name="Chest Xray",log_code=True)
# %%
def training(model, train_dataloader, num_epochs):
    optimizer_name = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
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
        
num_epochs =11 
training(model, train_dataloader, num_epochs)
experiment.end
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
      inputs = data[0].float().to(device),
      labels =  data[1].to(device)
      
      
      # Normalize the inputs
    #   inputs_m, inputs_s = inputs.mean(), inputs.std()
    #   inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)
      ic.ic(outputs)

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

# %%
end_ = 5
for i,data in enumerate(val_dataloader):
  
    if i < end_: 
        print(data)
        print(type(data))
        print(len(data))
    else: 
        break
# %%
model(inputs)

## 