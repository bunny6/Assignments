#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np       #importing libraries
import pandas as pd


# In[2]:


df=pd.read_csv("purchase_data.csv")   #reading file


# In[3]:


df.head()  #quick flash of dataset


# In[31]:


df.info()   #looking for the dtype of columns


# In[32]:


df.isnull().sum()  #looking for the null values in df


# In[34]:


df["SN"].value_counts()  # best way to check garbage values


# In[8]:


cat=[]                                 #just seperating categorical and numerical columns and saving them in list.
num=[]
for i in df:
    if df[i].dtype=="object":
        cat.append(i)
    else:
        num.append(i)


# In[35]:


cat


# In[36]:


num


# In[37]:


df.columns     #reading columns of df


# In[38]:


df.shape #total number of Items


# In[39]:


df


# In[41]:


df.shape #total number of players


# In[5]:


item_name_number=df["Item Name"].unique() #list of unique items


# In[54]:


item_name_number    #list of unique items


# In[52]:


unique_item=np.unique(df["Item Name"]) #number of unique items
print(len(unique_item))


# In[56]:


df["Price"].mean()  #average purchased price.


# In[57]:


df["Purchase ID"].count() #total number of purchase


# In[58]:


df["Price"].sum()  #total revenue


# In[60]:


df["Gender"].value_counts()  #total number of Male,Female,Others


# In[64]:


df["Gender"].value_counts(normalize=True)  #total number of male and females in percentage


# In[71]:


df.loc[0:20,["Gender",'Price']]


# In[73]:


filt=df["Gender"]=="Male"                      #purchase count of male
df.loc[filt]["Purchase ID"].value_counts().sum()


# In[74]:


filt=df["Gender"]=="Female"                    #purchase count of female
df.loc[filt]["Purchase ID"].value_counts().sum()


# In[75]:


filt=df["Gender"]=="Other / Non-Disclosed"        #purchase count of others
df.loc[filt]["Purchase ID"].value_counts().sum()


# In[76]:


filt1=df["Gender"]=="Male"                      #Average of total amount for male
df.loc[filt1]["Price"].mean()


# In[77]:


filt1=df["Gender"]=="female"                       #Average of total amount for female
df.loc[filt1]["Price"].mean()


# In[78]:


filt1=df["Gender"]=="Other / Non-Disclosed"                        #Average of total amount for others
df.loc[filt1]["Price"].mean()


# In[79]:


filt=df["Gender"]=="Male"                      #total purchase amount of male
df.loc[filt]["Price"].sum()


# In[80]:


filt=df["Gender"]=="Female"                      #total purchase amount of female
df.loc[filt]["Price"].sum()


# In[81]:


filt=df["Gender"]=="Other / Non-Disclosed"                      #total purchase amount of others
df.loc[filt]["Price"].sum()


# In[ ]:


filt=df["Gender"]=="Other / Non-Disclosed"                      #total purchase amount of others
df.loc[filt]["Price"].sum()


# In[ ]:




