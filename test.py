# %%
import os

# %%
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file

# %%
# The path to a pydicom test dataset
path = 'data/small_collection/009bc039326338823ca3aa84381f17f1.dicom'
ds = dcmread(path)
# `arr` is a numpy.ndarray
arr = ds.pixel_array

# %%
plt.imshow(arr, cmap="gray")
plt.show()

# %%
os.getcwd()
# %%


import sys
sys.path.append("/home/gyasis/Public/GD/Google Drive/Collection/ds-toolkit/custom_modules/")
import precheck
# %%
precheck.precheck('/home/gyasis/Public/GD/Google Drive/Collection/playground/audio')
# %%
import icecream as ic
# %%
import glob
# %%
glob.glob('/home/gyasis/Public/GD/Google Drive/Collection/playground/covid')
# %%
