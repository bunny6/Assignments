#!/usr/bin/env python
# coding: utf-8

# In[89]:


import PyPDF2


# In[90]:


pdfFileObj = open('sample.pdf', 'rb')


# In[91]:


pdfReader = PyPDF2.PdfFileReader(pdfFileObj)


# In[92]:


n=pdfReader.numPages


# In[93]:


print(n)


# In[94]:


textfile= open('textfile.txt','w') 


# In[95]:


for i in range(n):
    content=pdfReader.getPage(i).extractText()
    print(content)
    textfile.write(content)
    textfile.write('\n')
    
  
      
      
      
      


# In[96]:


file=open('textfile.txt','r')


# In[97]:


data = file.read()


# In[98]:


def searching_Word(sw):
    if sw in data:
        occurrences = data.count(sw)
        return occurrences
    else:
        return "Not Found Matched Word"
        
        
    
  
    
    
  
    
    
    
        
       
          
    
    
   


# In[99]:


sw=input("Enter the word to search")
print(searching_Word(sw))


# In[ ]:




