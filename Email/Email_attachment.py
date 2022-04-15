
#inbuilt library in python.
#importing library.
import smtplib  
from email.message import EmailMessage


msg=EmailMessage()
msg["Subject"]="Testing is being done"    #subject to sent.
msg["From"]="Shubham"                     #name of the sender.         
msg["To"]='dileepyadavneo@gmail.com'         #here we mention the email id of the receiver.
msg.set_content("Hello this mail is sent by using python")      #message which we have send.
with open("output.xlsx","rb") as f:                             #here we read the file which we have to send.
    file_data=f.read()
    file_name=f.name
    print("file name is",file_name)
    msg.add_attachment(file_data,maintype="application",subtype="xlsx",filename=file_name)     #subtype is the extension of the file.

server=smtplib.SMTP_SSL("smtp.gmail.com",465)                #establising the connection by providing the port number of gmail.
server.login("ghardes72@gmail.com","Shubham@12")             #email id which will be used to send the email.
server.send_message(msg)
server.quit()                                                #here we close the connection.





