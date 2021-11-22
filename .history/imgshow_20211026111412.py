# code for displaying multiple images in one figure 

# %% 
import glob
radimg = glob.glob('data/small_collection/*.dicom')
print(radimg)
# %%
import re
imagelist = []
for item in radimg:
    test = item.replace("data/small_collection\\","")
    print(test)
    imagelist.append(test)
    
print(len(imagelist))    
# %%
import numpy as np
from pydicom import dcmread

# %%
from matplotlib import pyplot as plt 

# create figure 
fig = plt.figure(figsize=(20, 40)) 
import matplotlib.patches as patches

def interest_area():
    
    # later get index for coordinates
    # Create a Rectangle patch
    rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    return ax.add_patch(rect)

# setting values to rows and column variables 
rows = 7
columns = 4

path = 'data/small_collection/'
i = 1
for item in imagelist:
    ds = dcmread(item)
    # `arr` is a numpy.ndarray
    arr = ds.pixel_array

    # Adds a subplot at the 1st position 
    fig.add_subplot(rows, columns, i) 
    # showing image 
    plt.imshow(arr, cmap="gray") 
    plt.axis('off') 
    i += 1


# %%
contents = dcmread(imagelist[0])
print(contents)
# %%
fig2 = plt.figure(figsize=(30,60))
arr = contents.pixel_array
plt.imshow(arr, cmap="gray")


# %%
# def interest_area():
    
#     # later get index for coordinates
#     # Create a Rectangle patch
#     rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

#     # Add the patch to the Axes
#     return fig2.add_patch(rect)

# fig2 = plt.figure(figsize=(30,60))
# interest_area()


# plt.imshow(arr, cmap='gray')

# %%
contents.Rows
# %%
contents[0x28, 0x30]
# %%
contents.patientssex
# %%
contents.HighBit
# %%
import pandas as pd 

df = pd.read_csv('data/train.csv')
df.head(10)

# %%
from torch.utils.data import DataLoader, Dataset, random_split

# %%
df = df[['image_id', 'class_id']]
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
#os.path.exists(df.imagepath[0])

# %%
df = df[['imagepath','class_id']]
# %%
df.head()
# %%


# %%
contents = dcmread(df.imagepath[0])
print(contents)
# %%
fig2 = plt.figure(figsize=(30,60))
arr = contents.pixel_array
plt.imshow(arr, cmap="gray")


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
        class_id = self.df.loc[index, 'class_id']
        return arr , class_id
        
    def __len__(self):
        return self.data_len

# %%
chestdata = MyDataset(df)
# %%
for i in range (4):
    # print(chestdata[i])
    print(chestdata[i][0])
    plt.imshow(chestdata[i][0], cmap="gray")
# %%

# %%
# %%


#import image and output as asarray
def input_