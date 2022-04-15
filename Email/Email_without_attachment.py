 #importing inbuilt library in python.
  
import smtplib 
from email.message import EmailMessage

msg=EmailMessage()
msg["Subject"]="Testing is being done"   #subject which we have to give.
msg["From"]="Shubham"                    #name of the sender
msg["To"]='sgharde14@gmail.com'          #here we mention the receivers email id.
msg.set_content("Hello this mail is sent by using python")       #the content we want to send.

server=smtplib.SMTP_SSL("smtp.gmail.com",465)           #establishing a connection with goggle gmail.
server.login("ghardes72@gmail.com","Shubham@12")         #here we mention the email id of the sender.
server.send_message(msg)
server.quit()                                           #here we close the connection.






