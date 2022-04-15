


import numpy as np       #importing libraries
import pandas as pd





df=pd.read_csv("purchase_data.csv")   #reading file




df.head()  #quick flash of dataset





df.info()   #looking for the dtype of columns


:


df.isnull().sum()  #looking for the null values in df





df["SN"].value_counts()  # best way to check garbage values





cat=[]                                 #just seperating categorical and numerical columns and saving them in list.
num=[]
for i in df:
    if df[i].dtype=="object":
        cat.append(i)
    else:
        num.append(i)





cat





num





df.columns     #reading columns of df





df.shape #total number of Items




df





df.shape #total number of players





item_name_number=df["Item Name"].unique() #list of unique items





item_name_number    #list of unique items





unique_item=np.unique(df["Item Name"]) #number of unique items
print(len(unique_item))





df["Price"].mean()  #average purchased price.




df["Purchase ID"].count() #total number of purchase




df["Price"].sum()  #total revenue


# In[60]:


df["Gender"].value_counts()  #total number of Male,Female,Others




df["Gender"].value_counts(normalize=True)  #total number of male and females in percentage




df.loc[0:20,["Gender",'Price']]





filt=df["Gender"]=="Male"                      #purchase count of male
df.loc[filt]["Purchase ID"].value_counts().sum()





filt=df["Gender"]=="Female"                    #purchase count of female
df.loc[filt]["Purchase ID"].value_counts().sum()





filt=df["Gender"]=="Other / Non-Disclosed"        #purchase count of others
df.loc[filt]["Purchase ID"].value_counts().sum()




filt1=df["Gender"]=="Male"                      #Average of total amount for male
df.loc[filt1]["Price"].mean()




filt1=df["Gender"]=="female"                       #Average of total amount for female
df.loc[filt1]["Price"].mean()





filt1=df["Gender"]=="Other / Non-Disclosed"                        #Average of total amount for others
df.loc[filt1]["Price"].mean()




filt=df["Gender"]=="Male"                      #total purchase amount of male
df.loc[filt]["Price"].sum()




filt=df["Gender"]=="Female"                      #total purchase amount of female
df.loc[filt]["Price"].sum()





filt=df["Gender"]=="Other / Non-Disclosed"                      #total purchase amount of others
df.loc[filt]["Price"].sum()





filt=df["Gender"]=="Other / Non-Disclosed"                      #total purchase amount of others
df.loc[filt]["Price"].sum()






