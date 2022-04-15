
#inbuilt library in python.
#importing library.
import smtplib  
from email.message import EmailMessage


msg=EmailMessage()
msg["Subject"]="Testing is being done"    #subject to sent.
msg["From"]="Shubham"                              
msg["To"]='dileepyadavneo@gmail.com'         #here we mention the email id of the receiver.
msg.set_content("Hello this mail is sent by using python")      #message which we have send.
with open("output.xlsx","rb") as f:
    file_data=f.read()
    file_name=f.name
    print("file name is",file_name)
    msg.add_attachment(file_data,maintype="application",subtype="xlsx",filename=file_name)

server=smtplib.SMTP_SSL("smtp.gmail.com",465)
server.login("ghardes72@gmail.com","Shubham@12")             #email id which will be used to send the email.
server.send_message(msg)
server.quit()





