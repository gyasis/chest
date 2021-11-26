# %%

try:
  %load_ext autotime
except:
  print("Console warning-- Autotime is jupyter platform specific")

# %%
import math
from pyforest import *
lazy_imports()
from pydicom import dcmread
# %%

df = pd.read_csv('/media/gyasis/Drive 2/Data/vinbigdata/train.csv')
df.head(10)

# %%
import seaborn as sns
sns.set_theme(style="dark")
sns.histplot(x=df.class_name, data=df)
plt.xticks(rotation=90)
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

# g1.set_xlabel("Class")
# g2.set_xlabel("Class")
g1.set_xticklabels(disease)
# %%

g = sns.histplot(x=y_train, order=disease)
plt.xticks(list(i for i in range(15)),disease, rotation=90)
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
def prepare_class_split(dataframe, target="class_name", p_split=0.30, test_target_split=0.50, verbose=True, helpers="dataframe"):
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
      print(outcomes_df.iat[i,1])
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

# %%
#use proposed_split to split the dataframe

def custom_split(dataframe,lable):
  class_list = list(dataframe[target].unique()
  for index, row in dataframe.iterrows():
    if row[lable] == class_list:
      print(row[lable])
      # if row[lable] == "OK!!":
      #   pass
      # else:
        # dataframe.drop(index, inplace=True)
# %%
custom_split(df,"class_name","OK!!")




# %% 
df
# %%
for index, row in df.iterrows():
  print(index)
  print(row)

# %%
count =113
for index, row in df.iterrows():
  
  if row["class_name"] == "Pneumothorax" and count > 0:
    print(row["class_name"])
    df.drop(index, inplace=True)
    count -=1
# %%
def custom_split(dataframe1,dataframe2):
  temp = list(dataframe1.index)
  for i, class_ in enumerate(temp):
    total = dataframe1.iat[i, 0]
    for index, row in dataframe2.iterrows():
      if row["class_name"] == class_ and total > 0 :
        total -= 1
        df.drop(index, inplace=True)
        
      else:
        print("Finished ", class_)
# %%
custom_split(test, df)
# %%
