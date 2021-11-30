# %% 
%load_ext autotime

# %% 
import numpy as np
import sklearn.model_selection as ms
import torchvision.datasets


# %%
import autoPyTorch
# %%
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
# %%
#count number of ones in a list of numbers 
def count_ones(list_of_numbers):
    np.sum(list_of_numbers == 1)