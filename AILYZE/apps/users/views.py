from django.shortcuts import render,HttpResponse,redirect
from apps.users.forms import UserChangePassword,RegisterUser,SummerizeType,SPecificQuestion,ThemeType,IdentifyViewpoint,CompareViewpoint,UplaodFileForm,ContactForm,ExcelForm,CategoriesForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.views import LogoutView
from django.urls import reverse_lazy
from django.views import View
from apps.users.models import User,UserQuery,Files
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth import update_session_auth_hash
from apps.users.enum import Anaylsis,Excelchoice
from django.views.generic.list import ListView
from django.views.generic import DetailView
from apps.users.utils import FileHandler,SumarrizeClass,QuestionClass,ThemeAnalysisClass,FrequencyHandlerClass,CompareViewPointsClass,ExcelTheme,ExcelCategories
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
import os
import pandas as pd
import openpyxl
from django.contrib import messages


class Home(View):
    def get(self,request):
        return render(request,'index.html')
    

@method_decorator(login_required, name='dispatch')
class UserProfile(View):
    def get(self,request):
        return render(request,'userprofile.html')
    

class LoginView(View):
    def get(self,request):
        fm=AuthenticationForm()
        return render(request,'login.html',{"form":fm})
    
    def post(self, request):
        obj = AuthenticationForm(request=request, data=request.POST)
        if obj.is_valid():
            email=obj.cleaned_data['username']
            password=obj.cleaned_data['password']
            user=authenticate(email=email,password=password)
            if user is not None:
                login(request,user)
                return redirect('user-profile')
        else:
            return render(request, 'login.html', {"form": obj})
        
class LogoutView(LogoutView):
    next_page=reverse_lazy("home")

class Register(View):
    def get(self,request):
        fm=RegisterUser()
        return render(request,'register.html',{"form":fm})
    
    def post(self,request):
        fm=RegisterUser(request.POST)
        if fm.is_valid():
            fm.save()
            return redirect('login')
        else:
            return render(request,'register.html',{'form':fm})

@method_decorator(login_required, name='dispatch')
class ChangePassword(View):
    def get(self,request):
        fm=UserChangePassword(user=request.user)
        return render(request,"changepassword.html",{'form':fm})
    
    def post(self,request):
        fm=UserChangePassword(user=request.user, data=request.POST)
        if fm.is_valid():
            user=fm.save()
            update_session_auth_hash(request,user)
            return redirect('user-profile')
        return render(request,"changepassword.html",{'form':fm})
    



class Getchoices(View):
    def get(self, request):
        form = UplaodFileForm()
        context = {'form': form}
        return render(request, "filedata.html", context)
    
    def post(self,request):
        file=request.FILES.getlist('file')
        for f in file:
            extension=os.path.splitext(f.name)[1]
            if extension.lower()  in ['.xlsx','.xls','.csv']:
                df=pd.read_excel(f)
                column_values=df.columns.values.tolist()
                request.session['column_values']=column_values
                choices=Excelchoice.choices()
            else:
                choices=Anaylsis.choices()
            return render(request, 'choices.html',{'choices':choices})
        return render(request,'filedata.html')
            

                





     

class UserQuestion(View):
    a = {
        Anaylsis.Summarize.value: lambda request,name: render(request,'chioceform.html',{'forms':SummerizeType(),"name":None}),
        Anaylsis.Ask_a_specific_question.value:  lambda request,name:render(request,'chioceform.html',{'forms':SPecificQuestion(),"name":None}),
        Anaylsis.Conduct_thematic_analysis.value: lambda request,name: render(request,'chioceform.html',{'forms':ThemeType(),"name":None}),
        Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:  lambda request,name: render(request,'chioceform.html',{'forms':IdentifyViewpoint(),"name":None}),
        Anaylsis.Compare_viewpoints_across_documents.value:  lambda request,name: render(request,'chioceform.html',{'forms':CompareViewpoint(),"name":None}),
        Excelchoice.Conduct_thematic_analysis_based_on_text_in_a_column.value:lambda request ,name:render(request,'chioceform.html',{'forms':ExcelForm(),'choices':request.session['column_values'],'name':name}),
        Excelchoice.Categorize_text_in_each_cell_in_a_column.value:lambda request, name:render(request,'chioceform.html',{'forms':CategoriesForm(),'choices':request.session['column_values'],'name':name})
    }
    def post(self, request):
        choice = request.POST.get('choice')
        request.session['choice']=choice
        render_fun = self.a.get(choice)
        if not render_fun:
            return render(request,'chioceform.html',{'forms':SummerizeType()}),
        name = 'Excel' if choice == Excelchoice.Conduct_thematic_analysis_based_on_text_in_a_column.value else 'ExcelCategory' if choice == Excelchoice.Categorize_text_in_each_cell_in_a_column.value else None
        return render_fun(request, name=name)


class ProcessQuery(View):
    def post(self, request):
        form = UplaodFileForm(request.POST, request.FILES)
        
        uploaded_file = request.FILES.getlist('file')
        upload_option = request.POST['upload_option']
        choice = request.session.get('choice')
        file_column=request.POST['file_column']
        if form.is_valid():
            if not isinstance(uploaded_file, list):
                uploaded_file = [uploaded_file]
            instance=FileHandler(uploaded_file)
            df=instance.upload_documents(max_documents=50,max_words=10000)

            if choice == Anaylsis.Summarize.value:
                forms = SummerizeType(request.POST)
                if forms.is_valid():
                    summary_type=forms.cleaned_data['summary']
                    summary_instruction=forms.cleaned_data['instruction']
                    instance=SumarrizeClass(is_demo=False,summary_type=summary_type,summary_instructions=summary_instruction,individual_summaries=False)
                    response=instance.call_summarize(df)
                    if request.user.is_authenticated:
                        UserQuery.objects.create(user=self.request.user,question={'Summary_type':summary_type,"Summary_instruction":summary_instruction},answer=response['summary'])                    
                        return redirect('/user-profile')
                    else:
                        UserQuery.objects.create(question={'Summary_type':summary_type,"Summary_instruction":summary_instruction},answer=response['summary'])
                        return render(request,'techdemo.html',{"response":response})       
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
            if choice==Anaylsis.Ask_a_specific_question.value:
                forms=SPecificQuestion(request.POST)
                if forms.is_valid():
                    question=forms.cleaned_data['question']
                    quesion_instruction=forms.cleaned_data['instruction']
                    question_keyword=forms.cleaned_data['keywords']
                    instance=QuestionClass(question=question,keywords=question_keyword,instruction=quesion_instruction)
                    response=instance.answer_question(df)     
                    if request.user.is_authenticated:     
                        UserQuery.objects.create(user=self.request.user,question={'question':question,"quesion_instruction":quesion_instruction,'question_keyword':question_keyword},answer=response)               
                        return redirect('/user-profile')
                    else:
                        data=UserQuery.objects.create(question={'question':question,"quesion_instruction":quesion_instruction,'question_keyword':question_keyword},answer=response)
                        return render(request,'techdemo.html',{"response":data})
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
            if choice==Anaylsis.Conduct_thematic_analysis.value:
                forms=ThemeType(request.POST)
                if forms.is_valid():
                    theme_type=forms.cleaned_data['theme_type']
                    instruction=forms.cleaned_data['instruction']
                    instance=ThemeAnalysisClass(theme_type=theme_type,theme_instruction=instruction)
                    response=instance.check_theme_type(df)
                    if request.user.is_authenticated:       
                        UserQuery.objects.create(user=self.request.user,question={'theme_type':theme_type,"instruction":instruction},answer=response['answer'])             
                        return redirect('/user-profile')
                    else:
                        UserQuery.objects.create(question={'theme_type':theme_type,"instruction":instruction},answer=response['answer'])
                        return render(request,'techdemo.html',{"response":response})
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
            
            if choice==Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:
                forms=IdentifyViewpoint(request.POST)
                if forms.is_valid():
                    instruction=forms.cleaned_data['instruction']
                    instance=FrequencyHandlerClass(frequency_viewpoint=instruction)
                    response=instance.call_frequency(df)
                    if request.user.is_authenticated:     
                        UserQuery.objects.create(user=self.request.user,question={" Frequency_instruction":instruction},answer=response['answer'])               
                        return redirect('/user-profile')
                    else:
                        UserQuery.objects.create(question={" Frequency_instruction":instruction},answer=response['answer'])
                        return render(request,'techdemo.html',{"response":response})
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
                
            if choice==Anaylsis.Compare_viewpoints_across_documents.value:
                forms=CompareViewpoint(request.POST)
                if forms.is_valid():
                    question=forms.cleaned_data['question']
                    instruction=forms.cleaned_data['instruction']
                    instruction_only=forms.cleaned_data['instruction_only']
                    keywords=forms.cleaned_data['keywords']
                    instance=CompareViewPointsClass(user=self.request.user,question_compare_groups=question,instruction_compare_groups=instruction,keywords_only=keywords,instructions_only=instruction_only)
                    response=instance.answer_question(df)
                    if request.user.is_authenticated:       
                        UserQuery.objects.create(user=self.request.user, question={" question":question,'instruction':instruction,'keywords':keywords,'instruction_only':instruction_only},answer=response)             
                        return redirect('/user-profile')
                    else:
                        UserQuery.objects.create(question={" question":question,'instruction':instruction,'keywords':keywords,'instruction_only':instruction_only},answer=response)
                        return render(request,'techdemo.html',{"response":response})
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
            
            if choice==Excelchoice.Conduct_thematic_analysis_based_on_text_in_a_column.value:
                choices=request.session['column_values']
                forms=ExcelForm(request.POST)
                forms.fields['file_column'].choices = [(choice, choice) for choice in choices] 
                if forms.is_valid():
                    file_column=forms.cleaned_data['file_column']
                    theme_type_excel=forms.cleaned_data['theme_type']
                    theme_instructions=forms.cleaned_data['theme_instructions']
                    instance=ExcelTheme(file_column=file_column,theme_type_excel=theme_type_excel,theme_instructions=theme_instructions,acesss=self.request.user.is_authenticated)
                    response=instance.excel_themes(df)
                    if request.user.is_authenticated:       
                        UserQuery.objects.create(user=self.request.user, question={" question":file_column,'theme_type_excel':theme_type_excel,'theme_instructions':theme_instructions},answer=response)             
                        return redirect('/user-profile')
                    else:
                        UserQuery.objects.create(question={" question":file_column,'theme_type_excel':theme_type_excel,'theme_instructions':theme_instructions},answer=response)
                        return render(request,'techdemo.html',{"response":response})
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
            
            if choice==Excelchoice.Categorize_text_in_each_cell_in_a_column.value:
                choices=request.session['column_values']
                forms=CategoriesForm(request.POST)
                forms.fields['file_column'].choices = [(choice, choice) for choice in choices] 
                if forms.is_valid():
                    file_column=forms.cleaned_data['file_column']
                    categories=forms.cleaned_data['categories']
                    categorize_instructions=forms.cleaned_data['categorize_instructions']
                    instance=ExcelCategories(file_column=file_column,categorize=categories,catgorize_instruction=categorize_instructions,access=self.request.user.is_authenticated)
                    response=instance.excel_categorize(df)
                    if request.user.is_authenticated:       
                        UserQuery.objects.create(user=self.request.user, question={" question":file_column,'categories':categories,'categorize_instructions':categorize_instructions},answer=response)             
                        return redirect('/user-profile')
                    else:
                        UserQuery.objects.create(question={" question":file_column,'categories':categories,'categorize_instructions':categorize_instructions},answer=response)
                        return render(request,'techdemo.html',{"response":response})
                messages.error(request,forms.errors)
                return render(request,'filedata.html')
                     
        else:   
            form = UplaodFileForm(request.POST, request.FILES)
        return render(request, 'filedata.html', {'form': form})



class About(View):
    def get(self, request):
        return render(request, "about.html")
 
class ShowData(ListView):
    model=UserQuery
    template_name='userdata.html'
    context_object_name='userdata'
    def get_queryset(self):
        queryset = super().get_queryset()
        user = self.request.user
        queryset = queryset.filter(user=user)
        return queryset

class DetailPage(View):
    def get(self,request,pk):
        obj = get_object_or_404(UserQuery, id=pk)
        data = {
            'question': obj.question,
            'answer': obj.answer,
        }
        return JsonResponse(data)
    
class Contactform(View):
    def get(self, request):
        form = ContactForm()
        return render(request, 'contact.html', {'form': form})

    def post(self, request):
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            form = ContactForm()  
            success_message = "Your form has been successfully submitted."  # Success message
            return render(request, 'contact.html', {'form': form, 'success_message': success_message})
        return render(request, 'contact.html', {'form': form})