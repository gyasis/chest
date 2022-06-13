# %%

try:
  %load_ext autotime
except:
  print("Console warning-- Autotime is jupyter platform specific")
# %%
from comet_ml import Experiment
import math
from pyforest import *
lazy_imports()

# %%



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

# torch.manual_seed(45)
torch.cuda._initialized = True

# %%

experiment = Experiment(api_key="xleFjfKO3kcwc56tglgC1d3zU",
                        project_name="Chest Xray",log_code=True)
# %%
import os.path
# %%
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

c_transform = nn.Sequential(transforms.Resize([256,]), 
                            transforms.CenterCrop(224),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
ten = torchvision.transforms.ToTensor()

# scripted_transforms = torch.jit.script(c_transform)
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
        # arr = scripted_transforms(arr)
        
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
ChestData_Valid = MyDataset(valid, transform=None)
ChestData_Test = MyDataset(test, transform=None)

# train_dataloader = DataLoader(train_ds, batch_size=128)
#  %%
ChestData_Aug_Train = AlbumentationsDataset(train, transform=transform)
ChestData_Visual = VisualDataset(train, transform=transform)
# %%
set_batchsize = 512
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


# additions 
# %%

# %%
from torch import nn as nn
num_classes = 15
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0")
model = nn.DataParallel(model, device_ids = [0,1])
model= model.to(device)


# %%
import copy
import matplotlib.pyplot as plt
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
        # print(rando)
        
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
    visualize_augmentations(ChestData_Visual, idx=5, samples=9, cols=3, save=False)   

# %%
# for t in range(9):
#     visualize_augmentations(ChestData_Valid, idx=5, samples=9,iterate='iterate', cols=3)

# # %%
# for t in range(9):
#     visualize_augmentations(ChestData_Aug_Train, idx=5, samples=9, cols=3)   

def tensor_to_cpu(x):
    for param in x:
        if torch.is_tensor(x[param]) == True:
            print("found tensor!")
            x[param] = x[param].cpu()
    return x

def save_model(model, optimizer, scheduler, model_name, num_epochs, loss, best_loss,
               save_dir='./models/'):
  model_dir = os.path.join(save_dir, model_name)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
    #   fix saving error when model is on cpu, here we make a copy to cpu to save
    #   device2 = torch.device("cpu")
    #   transfer_model = model.to(device2)
    #   transfer_optimizer = optimizer.to(device2)
    #   transfer_scheduler = scheduler.to(device2)
  
  model_path = os.path.join(model_dir, 'model.pt')
  optimizer_path = os.path.join(model_dir, 'optimizer.pt')
  scheduler_path = os.path.join(model_dir, 'scheduler.pt')
  loss_path = os.path.join(model_dir, 'loss.pt')
  best_loss_path = os.path.join(model_dir, 'best_loss.pt')
  
  #hopeful fix for save error : copy state_dicts and pull gpu tensors to cpu, and then save
  opt_dict = tensor_to_cpu(optimizer.state_dict())
  sched_dict = tensor_to_cpu(scheduler.state_dict())
  mod_dict = tensor_to_cpu(model.state_dict())
  
  print(scheduler.state_dict())
  print(sched_dict)

  torch.save(mod_dict, model_path)
  torch.save(opt_dict, optimizer_path)
  torch.save(sched_dict, scheduler_path)
  
  torch.save(loss, loss_path)
  torch.save(best_loss, best_loss_path)
  
  
  print('Checkpoint!/nSaved model to: {}'.format(model_path))

# %%
def training(model, train_dataloader, num_epochs):
    optimizer_name = torch.optim.SGD(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    weight = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
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
        
        
        for i, data in enumerate(tqdm(train_dataloader)):
            inputs = data[0].float().to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(optimizer)
            scheduler.step()
            
            
            
            # print(scheduler.state_dict())
            
            
            # optwrite = open("optimizer.pickle", "wb")
            # schedwrite = open("scheduler.pickle", "wb")
            # modwrite = open("model.pickle", "wb")
            
            # print("files are saved/nProgram will now end")
            # exit()
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
        save_model(model, optimizer, scheduler, 'chest_model', num_epochs, avg_loss, acc)
        
num_epochs = 2 



# %%
#save trained model with hyperparameters



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
# transformed = transform(image = arr)["image"]

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
# visualize(transformed)

def training(model, train_dataloader, num_epochs, save ='no'):
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
       
num_epochs = 3 
training(model, train_dataloader, num_epochs)

# %%
# %%
import optuna


def objective(trial):
  
  # Create the model and put it on the GPU if available
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter()
  torch.cuda.   `       _cache() 
  model = models.resnet18(pretrained=True)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)
  
  model = model.to(device)

  # Define the hyperparameters
  weight = torch.FloatTensor(weights).to(device)
  criterion = nn.CrossEntropyLoss(weight=weight)
  lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
  optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW','RMSprop', 'Adagrad'])
  num_epochs = trial.suggest_int('num_epochs', 5, 10 )
  optimizer = getattr(torch.optim, optimizer_name)(model.parameters(),lr=lr)
  
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, cycle_momentum=False,
                                                steps_per_epoch=int(len(train_dataloader)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
  
  # weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  
 
  # Repeat for each epoch
  for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(tqdm(train_dataloader)):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()
        
        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        running_acc = correct_prediction/total_prediction
        writer.add_scalar("Train/train_accuracy",running_acc, epoch)
        writer.add_scalar("Loss/train",loss, epoch)
        experiment.log_metric("Train/train_accuracy",running_acc, epoch)
        experiment.log_metric("Loss/train",loss, epoch)
        writer.flush()
        
        if i % 100 == 0:    # print every 10 mini-batches
          try:
            print('[%d, %5d] loss: %.3f  acc: %.2f' % (epoch + 1, i + 1, running_loss / i, running_acc))
          except ZeroDivisionError:
            print('division by zero')
            
    # Print stats at the end of the epoch
    num_batches = len(train_dataloader)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    experiment.log_metric("Accuracy", acc, epoch)
    writer.add_scalar("Train/epoch",epoch, epoch)
    writer.add_scalar("Train/loss",avg_loss, epoch)
    writer.add_scalar("Train/train_accuracy",acc, epoch)
    writer.flush()
    print('----------------------------------------------------------------')
    print('Pruning?')
    trial.report(acc, epoch)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
    print('----------------------------------------------------------------')
 
  return acc

sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize', pruner=optuna.pruners.PercentilePruner(n_startup_trials=5, n_warmup_steps=2, percentile=25.0))
study.optimize(objective, n_trials=10)
print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Trial number: ", trial.number)
print("  Loss (trial value): ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))  
















 # %%

#save trained model with hyperparameters
def save_model(model, optimizer, scheduler, model_name, epochs, loss, best_loss,
               save_dir='./models/'):
  model_dir = os.path.join(save_dir, model_name)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  model_path = os.path.join(model_dir, 'model.pt')
  optimizer_path = os.path.join(model_dir, 'optimizer.pt')
  scheduler_path = os.path.join(model_dir, 'scheduler.pt')
  loss_path = os.path.join(model_dir, 'loss.pt')
  best_loss_path = os.path.join(model_dir, 'best_loss.pt')
  torch.save(model.state_dict(), model_path)
  torch.save(optimizer.state_dict(), optimizer_path)
  torch.save(scheduler.state_dict(), scheduler_path)
  torch.save(loss, loss_path)
  torch.save(best_loss, best_loss_path)
  print('Saved model to: {}'.format(model_path))



def load_model(model, ):

 
 
 
 
# %%
 
#  opt_load = torch.load('./models/model_1/optimizer.pt')

opt_load = open("optimizer.pickle", "rb")
sched_load = open("scheduler.pickle", "rb")
model_load = open("model.pickle", "rb")

opt = pickle.load(opt_load)
sched = pickle.load(sched_load)
mod = pickle.load(model_load)
# %%
def tensor_to_cpu(x):
    for param in x:
        if torch.is_tensor(x[param]) == True:
            print("found tensor!")
            x[param] = x[param].cpu()
    return x
# %%
