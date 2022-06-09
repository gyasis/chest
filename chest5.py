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
from pydicom import dcmread
# %%

df = pd.read_csv('/media/gyasis/Drive 2/Data/vinbigdata/train.csv')
df.head(10)


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
# %%
# %%
import seaborn as sns
plt.xticks([i for i in range(0, len(df.class_id.unique()))],disease,rotation=90)
sns.set_theme(style="dark")
sns.histplot(x=(sorted(df.class_id)), data=df)

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
df1 = pd.get_dummies(df['class_id'])
sample_array = np.array(df1)
# %%
df1.head()
# %%
def get_class_frequencies():
    positive_freq = sample_array.sum(axis=0) / sample_array.shape[0]
    print(positive_freq)
    negative_freq = np.ones(positive_freq.shape) - positive_freq
    return positive_freq, negative_freq

p,n = get_class_frequencies()
# %%
plt.xticks([i for i in range(0, len(df.class_id.unique()))],disease,rotation=90)
data = pd.DataFrame({"Class": df1.columns, "Label": "Positive", "Value": p})
data = data.append([{"Class": df1.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(n)], ignore_index=True)

f = sns.barplot(x="Class", y="Value",hue="Label", data=data)
# %%
pos_weights = n
neg_weights = p
pos_contribution = p * pos_weights
neg_contribution = n * neg_weights
print(p)
print(n)
print("Weight to be added:  ",pos_contribution)

data = pd.DataFrame({"Class": df1.columns, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": df1.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
g = sns.barplot(x="Class", y="Value",hue="Label", data=data)
# %%
from sklearn.utils import compute_class_weight, compute_sample_weight
x  = compute_class_weight("balanced",sorted(df.class_id.unique()),df.class_id)

# %%
# show class weight
for i, proposed_weights in enumerate(x):
    # print(f"{df.class_name.unique()[i]}: {x[i]}")
    print('{:<25s}: {:<}'.format(disease[i], x[i]))
# %%
def weigh_and_show(dataframe):
  from sklearn.utils import compute_class_weight, compute_sample_weight
  x  = compute_class_weight("balanced",sorted(dataframe.class_id.unique()),dataframe.class_id)

  # show class weight
  for i, proposed_weights in enumerate(x):
      print('{:<25s}: {:<}'.format(disease[i], x[i]))
      
  # build dataframe with list of class_weights, class_name, sum of class names and product of class weights and sum of individual classes
  temp_weights = list(x)
  temp_class = list(sorted(dataframe.class_id.unique()))
  temp_class_name = disease
  temp_sum = list(dataframe.class_id.value_counts()[temp_class])
  temp_weight_products = [temp_weights[i] * temp_sum[i] for i in range(len(temp_weights))]
  
  # array check
  print(f"temp_weights: {len(temp_weights)}")
  print(f"class_name: {len(df.class_name.unique())}")
  print(f"sum_of_class_names: {len(temp_sum)}")
  
  #build dataframe
  temp_dataframe = pd.DataFrame({'class_weights': temp_weights, 
                                  'class_name': temp_class, 
                                  'sum_of_class_names': temp_sum, 
                                  'weight_products': temp_weight_products,})
                                 
  return temp_dataframe
# %%
def prepare_class_split(dataframe, target="class_id", p_split=0.20, test_target_split=0.50, verbose=False, helpers="dataframe"):
  dataframe = dataframe.copy()
  df_len = len(dataframe)
  class_amount = len(dataframe[target].unique())
  df_split = int(df_len * p_split)
  class_list = sorted(dataframe[target].unique())
  proposed_split = df_split/class_amount
  class_counts = dataframe[target].value_counts()
  class_counts = class_counts.sort_index()  
  
  outcomes = []
  total = []
  
  print("Total of Test Split is {} and Proposed split is {}".format(df_split,proposed_split))
  
  
  for i, lable in enumerate(class_list):
    percent_split = class_counts[lable] / df_len
    proposed_percent_split = class_counts[lable] / df_split
    total.append(class_counts[lable])
    if class_counts[lable] >= proposed_split * 2:
      if verbose == True:
        print(f"Class {disease[i]}:{lable} has {class_counts[lable]} instances, which is greater than the proposed split of {proposed_split}")
        print(f"Class {disease[i]}:{lable} has {percent_split} of the total data, which is greater than the proposed split of {proposed_percent_split}")
        print(f"Class {disease[i]}:{lable} is OK!!")
      outcomes.append("OK!!")
       
      
    elif class_counts[lable] < proposed_split * 2 and class_counts[lable] > proposed_split:
      if verbose == True:
        print("Class {} fails equity threshold, look to augment training dataset ".format(lable))
      outcomes.append("Augment??")
    elif class_counts[lable] < proposed_split:
      if verbose == True:
        print("Class {} fails equity threshold, look to remove training dataset ".format(lable))
        print("Class {} is {} and Proposed split is {}".format(lable,class_counts[lable],proposed_split))
        print(f"Class {lable} is less than the proposed split")
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
  
    if out == "Augment??":
      outcomes_df.iat[i,0] = outcomes_df.Split[i] * 0.80
    elif out == "Weights/Augment/Split!!":
      outcomes_df.iat[i, 0] = math.floor(outcomes_df.Set_Number[i]*0.50)
    elif out == "OK!!":
      pass
    else:
      print("Error")
  
  
  if helpers == "dataframe":
    return outcomes_df

# %%
analyze_split = prepare_class_split(df, target="class_id", p_split=0.30, test_target_split=0.50, verbose=False, helpers="dataframe")
# %%
analyze_split["class_name"] = disease
analyze_split = analyze_split[['class_name', 'Set_Number','Split','Outcome']]
analyze_split
# %%

def get_n_samples(dataframe, test):
    class_ids = sorted(dataframe['class_id'].unique())
    print(class_ids)
    samples = []
  
    for i in class_ids:
     
        n = test.Split[i]
        samples.extend(fast_random_sampling(dataframe, i, n))
        print(f"Class {i} : {test.class_name[i]} will have {n} samples")
        # print(i)
        # print(test.Split[i])
        print(f"{len(samples)} samples have been added")
    target_list = dataframe.index[[samples]]
    print(f"this is the length of samples: {len(samples)}")
    d2 = df.copy(deep=True)
    d3 = df.copy(deep=True)
    train_data = d2.drop(target_list, inplace=False)
    test_data = df.loc[df.index[target_list]]
    return train_data, test_data


def select_samples(df, class_id,n):
    random_rows = df.loc[df['class_id'] == class_id].sample(n=n)
    return random_rows.index.tolist()

def fast_random_sampling(df, class_id, n):
    df_class = df.loc[df['class_id'] == class_id]
    return np.random.default_rng().choice(df_class.index, n, replace=False).tolist()

# %%
train, test = get_n_samples(df, analyze_split)
# %%
train_dummies = pd.get_dummies(train.class_name)
test_dummies = pd.get_dummies(test.class_name)
# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.tick_params(axis='x',labelrotation=90)
ax2.tick_params(axis='x', labelrotation=90)
ax1.set_xticks([i for i in range(0, len(df.class_id.unique()))])
ax2.set_xticks([i for i in range(0, len(df.class_id.unique()))])
g1 = sns.histplot(train, ax=ax1)
g2= sns.histplot(test, ax=ax2)
g2.set_title("Test set")
g1.set_title("Train set")

# %%
def get_class_frequencies(dataframe,target):
    try:
        dataframe = pd.get_dummies(dataframe[target])
    except:
        dataframe = pd.get_dummies(dataframe[target])
        
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sample_array = np.array(dataframe)
    positive_freq = sample_array.sum(axis=0) / sample_array.shape[0]
    negative_freq = np.ones(positive_freq.shape) - positive_freq
    data = pd.DataFrame({"Class": dataframe.columns, "Label": "Positive", "Value": positive_freq})
    data = data.append([{"Class": dataframe.columns[l], "Label": "Negative", "Value": v} for l, v in enumerate(negative_freq)], ignore_index=True)
    plt.xticks([i for i in range(0, len(df[target].unique()))],disease,rotation=90)
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
weights = get_class_frequencies(train, "class_id")

# %%
test = test.sort_index()
test.head(20)
# %%
X, y = test[['class_name','imagepath']], test.class_id
X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
valid = pd.concat([X_valid, y_valid], axis=1, join='inner')
test = pd.concat([X_test, y_test], axis=1, join='inner')
# %%
train.head()
# %%
valid.head()
# %%
test.head()
# %%
analyzed = weigh_and_show(train)
# %%
analyzed
# %%
# from sklearn.preprocessing import MinMaxScaler  
# scaler = MinMaxScaler()
# print(scaler.fit(analyzed.class_weights))
# %%
weights = list(analyzed.class_weights)
# %%
weights
# %%
train = train[['imagepath','class_id']]
train = train.reset_index(drop=True)
valid = valid[['imagepath','class_id']]
valid = valid.reset_index(drop=True)
test = test[['imagepath','class_id']]
test = test.reset_index(drop=True)
# %%

# for idx, (data, image) in enumerate(tester):
#     print(idx)