#!/usr/bin/env python
# coding: utf-8

# In[131]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[132]:


pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)


# In[133]:


data = pd.read_excel('C:\\Users\\Mahdou\\Downloads\\dataset.xlsx')


# In[134]:


data


# # A) Exploratory Data Analysis :

# Objective:
# 
#  * Understand our data as best as possible
#  * Develop an initial modeling strategy

# Shape Analysis:
#  * target variable: SARS-Cov-2 exam result
#  * rows and columns: 5644, 111
#  * types of variables: qualitative: 70, quantitative: 41

# Analysis of missing values:
#  * a lot of missing values (several variables have more than 90 % of missing values)
#  * 2 groups of data 76% -> Viral test, 89% -> blood levels

# Visualization of the target:
# 
#  * 10% positive (558/5000)
#  * 90 % negative 

# Meaning of variables:
# 
#  * standardized continuous variables, skewed (asymmetric), blood test
#  * age quantile: difficult to interpret this graph, clearly this data has been processed, one could think 0-5, but this could also be a mathematical transformation. We can't know because the person who put this dataset doesn't specify it anywhere. But that's not very important !!!!
#  * qualitative variable: binary (0, 1), viral, Rhinovirus which seems very high

# Variables / Target relationship:
# 
#  * target / blood: the levels of Monocytes, Platelets, Leukocytes seem linked to covid-19 -> hypothesis to be tested
#  * target/age: young individuals are very little contaminated? -> be careful, we do not know the age, and we do not know when the dataset dates from (if it concerns children we know that children are affected as much as adults). On the other hand, this variable could be interesting to compare it with the results of blood tests.
#  * target / viral: dual diseases are very rare. Rhinovirus/Enterovirus positive - covid-19 negative? -> hypothesis to test? but it is possible that the region is experiencing an epidemic of this virus. In addition, we can very well have 2 viruses at the same time. All of this has no connection with covid-19

# # More detailed analysis:

# Relationship Variables / Variables:
# 
#  * blood_data / blood_data: certain variables are very correlated: +0.9 (to be monitored later)
#  * blood_data / age: very weak correlation between age and blood levels
#  * viral / viral: influenza rapid test gives poor results, it may be necessary to drop it
#  * disease / blood data relationship: Blood levels between patients and covid-19 are different
#  * hospitalization/is sick relationship:
#  * hospitalization / blood relationship: interesting in the case where we want to predict which department a patient should go to

# NaN analysis: viral: 1350(92/8), blood: 600(87/13), both: 90

# # Null hypotheses (H0):

#  * Individuals affected by covid-19 have significantly different levels of Leukocytes, Monocytes, Platelets
# 
#  * H0 = Mean rates are EQUAL in positive and negative individuals
#  *Individuals with any disease have significantly different rates

# # Data shape analysis :

# In[135]:


df = data.copy()


# In[136]:


df.shape


# In[137]:


df.dtypes


# In[138]:


df.dtypes.value_counts()


# we have 3 types of variables : float, object and int 

# In[139]:


df.dtypes.value_counts().plot.pie()


# In[140]:


df.isna()  ### True in a dataframe indicates a missing value,


# In[141]:


plt.figure(figsize=(23,12))
sns.heatmap(df.isna(), cbar=False)


# In[142]:


df.isna().sum()


# we note that for example 'patient id' have 0 missing values and others have more than 5600 !!!!!!!!

# In[143]:


df.isna().sum()/df.shape[0]  ### afficher ca en %


# In[144]:


(df.isna().sum()/df.shape[0]).sort_values(ascending=True)


# # Background Analysis:

# # 1. Initial visualization - Elimination of unnecessary columns

# In[145]:


df.isna().sum()/df.shape[0] < 0.9


# In[146]:


df.columns[df.isna().sum()/df.shape[0] < 0.9]


# In[147]:


df = df[df.columns[df.isna().sum()/df.shape[0] < 0.9]]
df.head()


# In[148]:


df.shape


# ### after removing columns who have more than 90 % of missing values, now we only have 39 columns 

# # Target column review :

# In[149]:


df['SARS-Cov-2 exam result']


# In[150]:


df['SARS-Cov-2 exam result'].value_counts(normalize=True)   #value_counts(normalize=True)


# we can see that 90% of target are negative 

# # histograms of continuous variables :

# In[151]:


for col in df.select_dtypes('float'):
    plt.figure()
    sns.histplot(df[col])    


# In[152]:


sns.histplot(df['Patient age quantile'], bins=20)


# In[153]:


df['Patient age quantile'].value_counts()


# # Qualitative Variables :

# In[154]:


for col in df.select_dtypes('object'):
    print(f'{col :-<50} {df[col].unique()}')


# In[155]:


for col in df.select_dtypes('object'):    
    plt.figure()
    df[col].value_counts().plot.pie()
    


# # Relation between Target / Variables

# ## Creation of positive and negative subsets :

# ### we create two subsets, one for the people tested positive and the other for the people tested negative

# In[156]:


pos_df = df[df['SARS-Cov-2 exam result'] == 'positive']


# In[157]:


neg_df = df[df['SARS-Cov-2 exam result'] == 'negative']


# In[158]:


pos_df


# In[159]:


neg_df


# ## Creation of the Blood and Viral sets :

# In[160]:


miss_val = df.isna().sum()/df.shape[0]


# In[161]:


miss_val


# In[162]:


blood_col = df.columns[(miss_val < 0.9) & (miss_val >0.88)]


# In[163]:


viral_col = df.columns[(miss_val < 0.88) & (miss_val > 0.75)]


# In[164]:


blood_col


# In[165]:


viral_col


# ## Relation between Target variable and Blood :

# In[166]:


for col in blood_col:
    plt.figure()
    sns.histplot(pos_df[col], label='positive')
    sns.histplot(neg_df[col], label='negative')
    plt.legend()


# ## Relation between Target variable and  age:

# In[167]:


sns.histplot(df['Patient age quantile'], label='age, target')


# ## Relation between Target  and Viral :

# In[168]:


pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A'])


# In[169]:


for col in viral_col:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')


# # A little more advanced analysis :

# ## Relation between Variables  and Variables :

# ### Blood / Blood :

# In[170]:


sns.pairplot(df[blood_col])


# In[171]:


sns.clustermap(df[blood_col].corr())


# ## Relation between Age and blood : 

# In[172]:


for col in blood_col:
    plt.figure()
    sns.lmplot(x='Patient age quantile', y=col, hue='SARS-Cov-2 exam result', data=df)


# In[173]:


#df.corr()['Patient age quantile'].sort_values()


# ## Relation between Influenza and rapid test :

# In[174]:


pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])


# In[175]:


pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])


# ## Relation between Viral and blood :

# # we create a new variable named "sick" :

# In[176]:


df[viral_col]


# In[177]:


df[viral_col[:-2]] == 'detected' ### return 'true' if detected, and 'false' otherwise


# In[178]:


df['sick'] = np.sum(df[viral_col[:-2]] == 'detected', axis=1) >=1 ### we create a new column named 'sick'
### True is treated as 1 and False as 0
### sick if has one 'true' at least


# In[179]:


df.head()


# ### we create two dataframe one for 'sick' and other for 'not sick'

# In[180]:


df_sick = df[df['sick'] == True]
df_notsick = df[df['sick'] == False]


# In[181]:


df_sick  ### we have all dataframe but only with sick people


# In[182]:


df_notsick  ### all dataframe but with not sick people 


# In[183]:


blood_col


# In[184]:


for col in blood_col:
    plt.figure()
    sns.histplot(df_sick[col], label='sick')
    sns.histplot(df_notsick[col], label='not sick')
    plt.legend()


# ### we are going to create now the 'hospitalization' function  to classify the hospitalization status of patients :

# In[185]:


def hospitalization(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'monitoring'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'semi intensive treatment'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'intensive care'
    else:
        return 'we dont know'


# In[186]:


df['the case'] = df.apply(hospitalization, axis=1)


# In[187]:


df.head()


# In[188]:


df['the case'].unique()


# In[189]:


for col in blood_col:
    plt.figure()
    for case in df['the case'].unique():
        sns.histplot(df[df['the case']==case][col], label=case)
    plt.legend()


# In[190]:


df[blood_col].count()


# In[191]:


df[viral_col].count()


# In[192]:


df1 = df[viral_col[:-2]]
df1['corona virus'] = df['SARS-Cov-2 exam result']
df1.dropna()['corona virus'].value_counts(normalize=True)


# In[193]:


### dropna() : removes rows with any NaN values by default.


# In[194]:


df2 = df[blood_col]
df2['corona virus'] = df['SARS-Cov-2 exam result']
df2.dropna()['corona virus'].value_counts(normalize=True)


# #### despite eliminating the columns containing the missing values, there remains almost the same proportions of sick and non-diseased cases

# # T-Test

# In[195]:


from scipy.stats import ttest_ind


# In[196]:


pos_df


# ## we will create a balanced dataset 

# ### The purpose of this line is to create a balanced dataset by ensuring that the number of negative cases matches the number of positive cases : 

# In[197]:


balanc_neg = neg_df.sample(pos_df.shape[0])


# In[198]:


balanc_neg


# In[199]:


balanc_neg.shape


# In[200]:


pos_df.shape


# In[201]:


def ind_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanc_neg[col].dropna(), pos_df[col].dropna())
    if p < alpha:
        return 'we reject the null hypothesis'
    else :
        return 'we do not reject the null hypothesis'


# In[202]:


for col in blood_col:
    print(f'{col :-<50} {ind_test(col)}')


# # B) PRE-PROCESSING :

# In[203]:


df.head()


# ## We create a subsets : 

# In[204]:


miss_val = df.isna().sum()/df.shape[0]


# In[205]:


miss_val


# In[206]:


blood_col = list(df.columns[(miss_val < 0.9) & (miss_val >0.88)])
viral_col = list(df.columns[(miss_val < 0.80) & (miss_val > 0.75)])


# In[207]:


blood_col


# In[208]:


viral_col


# In[209]:


key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']


# In[210]:


df = df[key_columns + blood_col + viral_col]
df.head()


# In[211]:


df.shape


# #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# In[212]:


code = {'negative':0,  ### we define the encoding mapping
            'positive':1,
            'not_detected':0,
            'detected':1}


# In[213]:


def encodiing(df):
    code = {'negative':0,  ### we define the encoding mapping
            'positive':1,
            'not_detected':0,
            'detected':1}
    
    for col in df.select_dtypes('object').columns:  ### we select names of object columns
        df.loc[:,col] = df[col].map(code) ## we applie our code by the 'map' function for all rows of dataframe with object col
        
    return df


# In[214]:


df = encodiing(df)


# In[215]:


df.head()


# # that's gooood, we encoded our dataframe !!!

# In[216]:


def feature_engineering(df):
    df['sick'] = np.sum(df[viral_col[:-2]] == 'detected', axis=1) >=1  ### creating the new column: 'sick'  
    df = df.drop(viral_col, axis=1)
    return df


# In[217]:


df = feature_engineering(df)


# In[218]:


df.head()


# In[219]:


df.shape


# In[ ]:





# In[ ]:





# In[220]:


def imputation(df):
    # Loop through each column and fill missing values with the mean of that column
    for col in df.columns:
        if df[col].isnull().any():  # Check if there are any missing values in the column
            df[col] = df[col].fillna(df[col].mean())
    return df


# In[221]:


df = imputation(df)


# In[222]:


df


# # good, now we should splitting our data in training and testing :

# In[223]:


df.head()


# In[224]:


import pandas as pd
from sklearn.model_selection import train_test_split

X = data.drop(columns='SARS-Cov-2 exam result')
y = data['SARS-Cov-2 exam result']


# In[225]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)


# In[226]:


train_set.shape


# In[227]:


test_set.shape


# ### So far, Alll right 

# In[228]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[229]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_set.drop('SARS-Cov-2 exam result', axis=1), train_set['SARS-Cov-2 exam result'])


# In[230]:


predictions = model.predict(test_set.drop('SARS-Cov-2 exam result', axis=1))


# In[231]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# In[232]:


print(test_set['SARS-Cov-2 exam result'].unique())


# In[233]:


predictions = model.predict(test_set.drop('SARS-Cov-2 exam result', axis=1))
print(np.unique(predictions))  


# # my predictions values are not in '0' or '1', we should solving this problem !!!!

# In[239]:


import numpy as np

# Assuming the output of your model is in 'predictions'
# Convert scores to binary labels using a threshold of 0
binary_predictions = (predictions >= 0.15).astype(int)

# Check the unique values in the binary predictions
print("Unique values in binary predictions:", np.unique(binary_predictions))


# In[240]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Convert the actual test labels to integers if they are not already
test_labels = test_set['SARS-Cov-2 exam result'].astype(int)

# Calculate and print evaluation metrics
accuracy = accuracy_score(test_labels, binary_predictions)
precision = precision_score(test_labels, binary_predictions)
recall = recall_score(test_labels, binary_predictions)
f1 = f1_score(test_labels, binary_predictions)
cm = confusion_matrix(test_labels, binary_predictions)
report = classification_report(test_labels, binary_predictions)

# Print results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Confusion Matrix:\n{cm}')
print(f'Classification Report:\n{report}')


# In[ ]:




