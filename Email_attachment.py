#!/usr/bin/env python
# coding: utf-8

# In[1]:


import smtplib  #inbuilt library in python
from email.message import EmailMessage


# In[2]:


msg=EmailMessage()
msg["Subject"]="Testing is being done"
msg["From"]="Shubham"
msg["To"]='dileepyadavneo@gmail.com'
msg.set_content("Hello this mail is sent by using python")
with open("output.xlsx","rb") as f:
    file_data=f.read()
    file_name=f.name
    print("file name is",file_name)
    msg.add_attachment(file_data,maintype="application",subtype="xlsx",filename=file_name)

server=smtplib.SMTP_SSL("smtp.gmail.com",465)
server.login("ghardes72@gmail.com","Shubham@12")
server.send_message(msg)
server.quit()


# In[ ]:




