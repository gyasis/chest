# %%

try:
  %load_ext autotime
except:
  print("Console warning-- Autotime is jupyter platform specific")

# %%
# %%
from comet_ml import Experiment
import math
from pyforest import *
lazy_imports()
from pydicom import dcmread
# %%

df = pd.read_csv('/media/gyasis/Drive 2/Data/vinbigdata/train.csv')
df.head(10)

# %%
import seaborn as sns
plt.xticks(rotation=90)
sns.set_theme(style="dark")
sns.histplot(x=df.class_name, data=df)

# %%
df=df[['class_name','class_id','image_id']]
# %%
def build_path(x):
    path_ = '/media/gyasis/Drive 2/Data/vinbigdata/train/'
    filetype = '.dicom'
    x = (path_+x+filetype)
    return x

df['imagepath'] = df['image_id'].apply(lambda x: build_path(x))
df=df[['class_name', 'class_id','imagepath']]
df.head()
# %%
pd.get_dummies(df['class_name'])

# %%
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
df1.columns = df1.columns.astype("int").map(lambda x: disease[x])
sample_array = np.array(df1)

# %%
def get_class_frequencies():
  positive_freq = sample_array.sum(axis=0) / sample_array.shape[0]
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

from sklearn.model_selection import train_test_split

X, y = df.imagepath, df.class_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
print(f"Numbers of train instances by class: {np.bincount(y_train)}")
print(f"Numbers of test instances by class: {np.bincount(y_test)}")
# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.tick_params(axis='x', labelrotation=90)
ax2.tick_params(axis='x', labelrotation=90)
g1 = sns.histplot(x=(list(map(lambda y_train: disease[y_train], y_train))), ax=ax1 )
g2= sns.histplot(x=(list(map(lambda y_test: disease[y_test], y_test))), ax=ax2, )
g2.set_title("Test set")
g1.set_title("Train set")

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# %%
print(len(df))
# %%
print(len(df)/len(df.class_id.unique()))
# %%
proposed_split = (len(df)*0.20)
# %%
proposed_split / len(df.class_id.unique())
# %%
print(len(df.class_id.unique()))
# %%
def prepare_class_split(dataframe, target="class_name", p_split=0.30, test_target_split=0.50, verbose=False, helpers="dataframe"):
  dataframe = dataframe.copy()
  df_len = len(dataframe)
  class_amount = len(dataframe[target].unique())
  df_split = int(df_len * p_split)
  class_list = list(dataframe[target].unique())
  
  proposed_split = df_split/class_amount
  
  class_counts = dataframe[target].value_counts()
  # print(df_len,df_split,proposed_split,class_counts)
  
  outcomes = []
  total = []
  
  print("Total of Test Split is {} and Proposed split is {}".format(df_split,proposed_split))
  
  
  for lable in class_list:
    percent_split = class_counts[lable] / df_len
    proposed_percent_split = class_counts[lable] / df_split
    total.append(class_counts[lable])
    if class_counts[lable] >= proposed_split * 2:
      if verbose == True:
        print(f"Class {lable} has {class_counts[lable]} instances, which is greater than the proposed split of {proposed_split}")
        print(f"Class {lable} has {percent_split} of the total data, which is greater than the proposed split of {proposed_percent_split}")
        print("Class {} is OK!!".format(lable))
      outcomes.append("OK!!")
       
      
    elif class_counts[lable] < proposed_split * 2 and class_counts[lable] > proposed_split:
      if verbose == True:
        print("Class {} fails equity threshold, look to augment training dataset ".format(lable))
      outcomes.append("Augment??")
    elif class_counts[lable] < proposed_split:
      if verbose == True:
        print("Class {} fails equity threshold, look to remove training dataset ".format(lable))
        print("Class {} is {} and Proposed split is {}".format(lable,class_counts[lable],proposed_split))
        print("Class " + lable + " is less than the proposed split")
        print("Class {} is {} and the proposed split is {}".format(lable,class_counts[lable],proposed_split))
        print("Both augmentation and weights may be necessary!!")
      outcomes.append("Weights/Augment/Split!!")
  
  outcomes_df = pd.DataFrame()
  outcomes_df["Class"] = class_list
  outcomes_df["Split"] = math.floor(proposed_split)
  outcomes_df["Set_Number"] = total
  outcomes_df["Outcome"] = outcomes
  outcomes_df.set_index("Class", inplace=True)
  
  
  # Change dataframe based on outcomes
  for i, out in enumerate(outcomes_df.Outcome):
    print(i)

    if out == "Augment??":
      outcomes_df.iat[i,0] = outcomes_df.Split[i] * 0.80
      # print(outcomes_df.iat[i,1])
    elif out == "Weights/Augment/Split!!":
      outcomes_df.iat[i, 0] = math.floor(outcomes_df.Set_Number[i]*0.50)
      # outcomes_df = outcomes_df.append(outcomes_df.at[i,"Set Number"] == math.floor(temp/0.5))
      # dataframe = dataframe.append(dataframe.loc[dataframe[target] == i])
    elif out == "OK!!":
      pass
    else:
      print("Error")
  
  
  if helpers == "dataframe":
    return outcomes_df


# %%
test = prepare_class_split(df, target="class_name", p_split=0.20, test_target_split=0.50, verbose=True, helpers="dataframe")
df['class'] = df['class_id'].apply(lambda x:disease[x])

# %%
def custom_split(dataframe1,dataframe2):
  dataframe2 = dataframe2.copy(deep=True)
  dataframe3 = dataframe2.copy(deep=True)
  dataframe2 = dataframe2.sample(frac=1)
  
  
  test_idx = []
  temp = list(dataframe1.index)
  print(temp)
  for i, class_ in enumerate(temp):
    total = dataframe1.iat[i, 0]
    print(total)
    for index, row in dataframe2.iterrows():
      if row["class_name"] == class_ and total > 0 :
        total -= 1
        test_idx.append(index)
        dataframe2.drop(index, inplace=True)
        
        # print("drop")
    print("Finished ", class_)
    
  print(len(dataframe2))
  
  dataframe3 = dataframe3.loc[dataframe3.index[test_idx]]
  dataframe2 = dataframe2.sample(frac=1)
  dataframe3 = dataframe3.sample(frac=1)
  print(len(dataframe3))
  return dataframe2, dataframe3
# %%
train_df, test_df= custom_split(test, df)
# %%
train_df.head()
# %%
train_df["class_name"].value_counts()

#%%
test_df.head()

# %%
test_df["class_name"].value_counts()

# %%
train_dummies = pd.get_dummies(train_df.class_name)
test_dummies = pd.get_dummies(test_df.class_name)

# %%
train_dummies.head(20)

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.tick_params(axis='x', labelrotation=90)
ax2.tick_params(axis='x', labelrotation=90)
g1 = sns.histplot(train_df, ax=ax1)
g2= sns.histplot(test_df, ax=ax2)
g2.set_title("Test set")
g1.set_title("Train set")
# g1.set_xticklabels(disease)


# %% 
def get_class_frequencies(dataframe,target):
  try:
    dataframe = pd.get_dummies(dataframe[target].astype(str))
  except:
    dataframe = pd.get_dummies(dataframe[target])
    
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
  sample_array = np.array(dataframe)
  positive_freq = sample_array.sum(axis=0) / sample_array.shape[0]
  negative_freq = np.ones(positive_freq.shape) - positive_freq
  data = pd.DataFrame({"Class": dataframe.columns, "Label": "Positive", "Value": positive_freq})
  data = data.append([{"Class": dataframe.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(negative_freq)], ignore_index=True)
  plt.xticks(rotation=90)
  sns.barplot(x="Class", y="Value",hue="Label", data=data, ax=ax1)
  pos_weights = negative_freq
  neg_weights = positive_freq
  pos_contribution = positive_freq * pos_weights
  neg_contribution = negative_freq * neg_weights

  # print("Weight to be added:  ",pos_contribution)
  
  data1 = pd.DataFrame({"Class": dataframe.columns, "Label": "Positive", "Value": pos_contribution})
  data1 = data1.append([{"Class": dataframe.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(neg_contribution)], ignore_index=True)
  ax1.tick_params(axis='x', labelrotation=90)
  ax2.tick_params(axis='x', labelrotation=90)
  sns.barplot(x="Class", y="Value",hue="Label", data=data1, ax=ax2)
  
  return pos_contribution


# %%
weights = get_class_frequencies(df, "class_name")
# %%

X, y = test_df.imagepath, test_df.class_id
X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
# %%
Valid_df = pd.concat([X_valid, y_valid], axis=1, join='inner')
Test_df = pd.concat([X_test, y_test], axis=1, join='inner')
Train_df = train_df[['imagepath', 'class_id']]

# %%

import os
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import icecream as ic
from IPython.display import  display
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from pydicom import dcmread


torch.cuda._initialized = True

# %%

experiment = Experiment(api_key="xleFjfKO3kcwc56tglgC1d3zU",
                        project_name="Chest Xray",log_code=True)
# %%
# %%
import os.path


# %%



# %%

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
                            A.RandomBrightnessContrast(p=0.1, contrast_limit=0.005, brightness_limit=0.005,),
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

W_o_ten_transform = A.Compose(
    [A.Resize(width=256,height=256, always_apply=True),
                       A.HorizontalFlip(p=0.5),
                       A.OneOf([
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.25),
                            # A.RandomBrightnessContrast(p=0.1, contrast_limit=0.05, brightness_limit=0.05,),
                            A.InvertImg(p=0.02),
                       ]),
                       A.OneOf([
                           A.RandomCrop(width=224, height=224, p=0.5),
                           A.CenterCrop(width=224, height=224, p=0.5),
                           
                       ]),
                       A.Resize(width=224, height=224, always_apply=True),
                       A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    #    ToTensorV2()
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
      
class VisualDataset(Dataset):
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
        arr = W_o_ten_transform(image = arr)["image"]
        # print(f"this is the index {index}") 
        # print(f"this is the the df index {self.df.index[index]}")
        return arr, class_id, 
    

# %%
ChestData_Valid = MyDataset(Valid_df, transform=None)
ChestData_Test = MyDataset(Test_df, transform=None)

# train_dataloader = DataLoader(train_ds, batch_size=128)
#  %%
ChestData_Aug_Train = AlbumentationsDataset(Train_df, transform=transform)
ChestData_Visual = VisualDataset(Train_df, transform=transform)
# %%
set_batchsize = 128
# %%
from torch.utils.data import DataLoader, Dataset, random_split
# num_items = len(ChestData_Aug)
# num_train = round(num_items * 0.7)
# num_val = num_items - num_train
# train_ds, val_ds = random_split(ChestData_Aug, [num_train, num_val])
train_dataloader = DataLoader(ChestData_Aug_Train, batch_size=set_batchsize, num_workers=4, pin_memory=True, shuffle=True)
val_dataloader = DataLoader(ChestData_Valid,batch_size=set_batchsize, num_workers=4, pin_memory=True,shuffle=False)


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
import copy

# %%

# get third element of  each tuple in a list of tuples
def get_third_element(tup):
      
    return tup[::2]



# %%
def visualize_augmentations(dataset, idx=12,iterate='random', samples=2, cols=2, save=False):
    dataset = copy.deepcopy(dataset)
    dataset.transform = transform
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
   
 
    for i in range(samples):
        # rando = random.randint(0,len(dataset)-1)
        rando = np.random.randint(0,len(dataset)-1)
        print(rando)
        
        while True:
          try:
            if (iterate=='random'):
                image, _ = dataset[rando]
                ax.ravel()[i].imshow(image[:,:,0],cmap='gray')
                ax.ravel()[i].set_axis_off()
                break
            else:
                image, _ = dataset[i]
                ax.ravel()[i].imshow(image[0,:,:],cmap='gray')
                ax.ravel()[i].set_axis_off()
                break
              
          except:
              print("possible index error")
              rando = np.random.randint(0,len(dataset)-1)
              
              
    if save == True:
      plt.tight_layout()
      filename = 'augmented_images_' + str(rando) + '.png'
      plt.savefig(filename)
      experiment.log_image(image_data = filename) 
      plt.show()
    
    
# %%
for t in range(9):
    visualize_augmentations(ChestData_Visual, idx=5, samples=9, cols=3)   

# %%
# for t in range(9):
#     visualize_augmentations(ChestData_Valid, idx=5, samples=9,iterate='iterate', cols=3)

# # %%
# for t in range(9):
#     visualize_augmentations(ChestData_Aug_Train, idx=5, samples=9, cols=3)   

# %%
def training(model, train_dataloader, num_epochs):
    optimizer_name = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(pos_contribution).type(torch.FloatTensor).to(device))
    optimizer = optimizer_name
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        
        
        for i, data, in enumerate(tqdm(train_dataloader), leave=False):
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
        
num_epochs = 6 

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


transformed = transform(image = arr)["image"]

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
visualize(transformed)

def training(model, train_dataloader, num_epochs):
    optimizer_name = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_name
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.01,
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
                
        num_batches = len(train_dataloader)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
       
num_epochs = 6 

training(model, train_dataloader, num_epochs)





# this function takes a datset, list of columns to augment, by a factor, and a savepath
# it will augment the dataset by the factor, and save the augmented images to the savepath
# then returns the augmented dataset as a dataframe

def augment_data(dataset, listoftoaugmentclass, numofaugmentation ,savepath):
    path = []
    class = []
    for i in listoftoaugmentclass:
        print(i)
        for j in random.sample(dataset[j][1], len(dataset)):
            if dataset[j][1] == i:
                print(dataset[j][1])
                ds = dcmread(dataset[j][0])
                arr = ds.pixel_array
                arr = arr.astype('float')
                arr = np.stack((arr,)*3, axis=-1)
                transformed = transform(image = arr)["image"]
                savepath = savepath + str(i) + '/'
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                plt.imsave(savepath + str(j) + '.png', transformed)
                path.append(savepath + str(j) + '.png')
                class.append(i)
        
    df = pd.DataFrame(list(zip(path,class)),columns=['imagepath','class'])
    return df

# for loop in range of lenth of list but random interation
def loop_random(len(df1)):
  
  
  
Analyze
Create
Augment
Train
  Initialize
  Tune
Validate
