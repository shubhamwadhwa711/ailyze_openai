


from django.core.mail import send_mail
from django.conf  import settings

def Contact_email(email):
    subject = "Contact"
    message = ""
    email_from = settings.EMAIL_HOST
    send_mail(subject,message, email_from,[email])
    print("Message Sent")
   