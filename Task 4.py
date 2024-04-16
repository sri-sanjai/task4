#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import stopwords

import string

from nltk import pos_tag

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import seaborn as sns

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import nltk
nltk.download('stopwords')


# In[7]:


data=pd.read_csv('twitter_training.csv')


# In[8]:


v_data=pd.read_csv('twitter_validation.csv')


# In[9]:


data


# In[10]:


v_data


# In[11]:


data.columns =['id','game','sentiment','text']
v_data.columns =['id','game','sentiment','text']


# In[12]:


data


# In[13]:


v_data


# In[14]:


data.shape


# In[16]:


data.columns


# In[17]:


data.describe(include='all')


# In[19]:


id_types =data['id'].value_counts()
id_types


# In[20]:


plt.figure(figsize=(12,7))
sns.barplot(y=id_types.index,x=id_types.values)
plt.xlabel('type')
plt.ylabel('count')
plt.title('# of Id vs Count')
plt.show()


# In[22]:


game_types = data['game'].value_counts()
game_types


# In[24]:


plt.figure(figsize=(14,10))
sns.barplot(x=game_types.values,y=game_types.index)
plt.title('# of Games and their count')
plt.ylabel('type')
plt.xlabel('count')
plt.show()


# In[25]:


sns.catplot(x="game",hue="sentiment",kind="count",height=10,aspect=3,data=data)


# In[30]:


total_null=data.isnull().sum().sort_values (ascending=False)

percent= ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False) 
print("Total records", data.shape[0])

missing_data = pd.concat([total_null, percent.round(2)], axis=1, keys=['Total Missing', 'In Percent'])
missing_data.head(10)


# In[35]:


data.dropna(subset= 'text', inplace=True)



# In[42]:


total_null=data.isnull().sum().sort_values(ascending= False) 
percent =((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending =False)
print("Total records=", data.shape[0])
missing_data = pd.concat([total_null, percent.round(2)], axis=1, keys=["Total Missing", "In Percent"])
missing_data.head(10)


# In[46]:


data.dropna(subset=['text'],inplace=True)


total_null=data.isnull().sum().sort_values(ascending=False)
percent= ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending =False)
print("Total records = ", data.shape[0])
missing_data= pd.concat([total_null, percent.round(2)],axis=1, keys =['Total Missing', 'In Percent'])
missing_data.head(10)


# In[48]:


train0=data[data['sentiment']=="Negative"]
train1=data[data['sentiment']=="Positive"]
train2=data[data['sentiment']=="Irrelevant"]
train3=data[data['sentiment']=="Neutral"]


# In[49]:


train0.shape,train1.shape,train2.shape, train3.shape


# In[50]:


train0=train0[:int(train0.shape[0]/12)]
train1=train1[:int(train1.shape[0]/12)]
train2=train2[:int(train2.shape[0]/12)]
train3=train3[:int(train3.shape[0]/12)]


# In[51]:


train0.shape,train1.shape,train2.shape, train3.shape


# In[53]:


data=pd.concat([train0, train1, train2, train3], axis=0)
data


# In[54]:


id_types=data['id'].value_counts()
id_types


# In[57]:


plt.figure(figsize=(12, 7))
sns.barplot(x=id_types.values, y=id_types.index)

plt.xlabel('Type')
plt.ylabel('Count')

plt.title('# of TV shows vs Movies')

plt.show()


# In[60]:


game_types = data['game'].value_counts()
game_types


# In[61]:


plt.figure(figsize=(12, 7))
sns.barplot(x=game_types.values, y=game_types.index)

plt.xlabel('Type')
plt.ylabel('Count')

plt.title('# of TV shows vs Movies')

plt.show()


# In[63]:


sentiment_types =data['sentiment'].value_counts()
sentiment_types


# In[64]:


plt.figure(figsize=(12,7))
plt.pie(x=sentiment_types.values,labels=sentiment_types.index,autopct='%.1f%%', explode=[0.1,0.1,0,0])
plt.title('The Difference in the Type of Contents')
plt.show()


# In[65]:


sns.catplot(x='game',hue='sentiment',kind='count',height=7,aspect=2,data=data)


# In[66]:


from sklearn import preprocessing

label_encoder  = preprocessing.LabelEncoder()


# In[67]:


data['sentiment']=label_encoder.fit_transform(data['sentiment'])

data['game']=label_encoder.fit_transform(data['game'])
v_data['sentiment']=label_encoder.fit_transform(v_data['sentiment'])
v_data['game']=label_encoder.fit_transform(v_data['game'])


# In[69]:


data= data.drop(['id'], axis=1)

data


# In[70]:


data.nunique()


# In[71]:


v_data.nunique()


# In[ ]:




