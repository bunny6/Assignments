#!/usr/bin/env python
# coding: utf-8

# In[1]:


import smtplib  #inbuilt library in python
from email.message import EmailMessage


# In[13]:


msg=EmailMessage()
msg["Subject"]="Testing is being done"
msg["From"]="Shubham"
msg["To"]='sgharde14@gmail.com'
msg.set_content("Hello this mail is sent by using python")

server=smtplib.SMTP_SSL("smtp.gmail.com",465)
server.login("ghardes72@gmail.com","Shubham@12")
server.send_message(msg)
server.quit()


# In[ ]:




