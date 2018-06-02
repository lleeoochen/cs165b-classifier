import os
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

#Setup constants
PASSWORD = os.environ.get('TESTBOT_PASSWORD')
FROM_EMAIL = "testbotchen@gmail.com"
TO_EMAIL = "lleeoochen@gmail.com"

#Condition check
if PASSWORD == None:
    print ('Password not found. Please set to environment variable TESTBOT_PASSWORD')
    exit()

#Setup email title
msg = MIMEMultipart()
msg['From'] = FROM_EMAIL
msg['To'] = TO_EMAIL
msg['Subject'] = "Deep Learning is Done! [From CSIL]"

#Setup email message
body = "Dear Wei Tung,\n\nQuickly open your csil and check. Good luck and have a nice day!\n\nSincerely,\nYour Lovely Tester Bot"
msg.attach(MIMEText(body, 'plain'))

#Send email
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(FROM_EMAIL, PASSWORD)
text = msg.as_string()
server.sendmail(FROM_EMAIL, TO_EMAIL, text)
server.quit()
 
