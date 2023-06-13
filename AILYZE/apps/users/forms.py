from django.contrib.auth.forms import PasswordChangeForm,UserCreationForm,AuthenticationForm
from apps.users.models import User,Files,ContactModel
from django import forms
from django.utils.translation import gettext_lazy as _
from apps.users.enum import Anaylsis
from multiupload.fields import MultiFileField
import pandas as pd
import openpyxl
from django.forms.widgets import ClearableFileInput, FILE_INPUT_CONTRADICTION
from multiupload.fields import MultiFileField


class RegisterUser(UserCreationForm):
    username=forms.CharField(widget=forms.TextInput(attrs={
        'class':"form-control"  
    }))
    email=forms.EmailField(widget=forms.EmailInput(attrs={
        'class':"form-control"  
    }))
    password1=forms.CharField(widget=forms.PasswordInput
                              (attrs={
        'class':"form-control"  
    }))
    password2=forms.CharField(widget=forms.PasswordInput
                              (attrs={
        'class':"form-control"  
    }))
    class Meta:
        model=User
        fields=['username','email','password1','password2']


class UserChangePassword(PasswordChangeForm):
    old_password=forms.CharField(label="Old Password" ,widget=forms.PasswordInput
                              (attrs={
        'class':"form-control" ,"placeholder":" Enter your old  password ",
    }))
    new_password1=forms.CharField(label="New Password" ,widget=forms.PasswordInput
                              (attrs={
        'class':"form-control","placeholder":" Enter your new pasword",  
    }))
    new_password2=forms.CharField(label="Re-enter New Password", widget=forms.PasswordInput
                              (attrs={
        'class':"form-control","placeholder":" Re-Enter your password",  
    }))
    class Meta:
        model=User
        fields=['old_password','new_password1','new_password2']




class UplaodFileForm(forms.ModelForm):
    UPLOAD_CHOICES = [
        ('new', 'Upload New File'),
        ('previous', 'Upload Files from Previous Uploads'),
    ]
    upload_option = forms.ChoiceField(choices=UPLOAD_CHOICES, widget=forms.Select())
    file = MultiFileField(min_num=1, max_num=10, max_file_size=1024*1024*5)
    # file = forms.FileField(widget=MultipleFileInput(attrs={
    #     'multiple':True
    # }), required=False)
    email = forms.EmailField(required=False)
    class Meta:
        model = Files
        fields = ['upload_option', 'file', 'email']






class ContactForm(forms.ModelForm):
    class Meta:
        model=ContactModel
        fields=['name','email','message','subject']


        
class SummerizeType(forms.Form):
    SUMMARY_CHOICES = [
        ('Essay', 'Essay'),
        ('Bullet points', 'Bullet points'),
    ]
    summary=forms.ChoiceField(choices=SUMMARY_CHOICES ,widget=forms.Select(attrs={
        'class':'form-select'
    }))
    instruction=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response from chatbot (e.g., length, style)','class':'form-control'}
    ))




class SPecificQuestion(forms.Form):
    question=forms.CharField(required=True,error_messages={
        'required': 'Please enter your question option.'},widget=forms.Textarea(
        attrs={'placeholder':'Enter your Question','class':'form-control'}
    ))
    instruction=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response from chatbot (e.g., length, style)','class':'form-control'}
    ))
    keywords=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Keywords or section names to help chatbot extract relevant parts of the document to analyze','class':'form-control'}
    ))

class ThemeType(forms.Form):
    SUMMARY_CHOICE = [
        ('Codebook', 'Codebook'),
        ('Essay', 'Essay'),
    ]
    theme_type=forms.ChoiceField(choices=SUMMARY_CHOICE,widget=forms.Select(attrs={
        'class':'form-select'}))
    instruction=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response from chatbot (e.g., length, style):','class':'form-control'}
    ))

class IdentifyViewpoint(forms.Form):
    instruction=forms.CharField(required=True,widget=forms.Textarea(
        attrs={'placeholder':'Participants generally trust medical professionals about the flu and vaccine','class':'form-control'}
    ))


class CompareViewpoint(forms.Form):
    question=forms.CharField(required=True,error_messages={
        'required': 'Please enter your question option.'},widget=forms.Textarea(
        attrs={'placeholder':'Input your Question','class':'form-control'}
    ))
    instruction=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Participant 1, 2, 3, 4, 5 and 6 are male. Participants 7, 8, 9, 10, 11 and 12 are female.','class':'form-control'}
    ))
    instruction_only=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response from chatbot (e.g., length, style)','class':'form-control'}
    ))
    keywords=forms.CharField(required=False,widget=forms.Textarea(
        attrs={'placeholder':'Keywords or section names to help chatbot extract relevant parts of the document to analyze','class':'form-control'}
    ))






class ExcelForm(forms.Form):
    excel_choices=[
        ('Codebook', 'Codebook'),
        ('Essay', 'Essay'),
    ]
    file_column=forms.ChoiceField(choices=[])
    theme_type=forms.ChoiceField(choices=excel_choices)
    theme_instructions=forms.CharField(required=False,widget=forms.Textarea(attrs={
        'placeholder':'Additional instructions regarding your desired response.Focus on contradicting views. Reply in French'}
    ))

class CategoriesForm(forms.Form):
    file_column=forms.ChoiceField(choices=[])
    categories=forms.CharField(required=False,widget=forms.Textarea(attrs={
        'placeholder':'Specify the categories (separated by commas),  happy, neutral, sad'
    }))
    categorize_instructions=forms.CharField(required=False,widget=forms.Textarea(attrs={
        'placeholder':'Additional instructions regarding your desired response.Focus on contradicting views. Reply in French'
    }))
  