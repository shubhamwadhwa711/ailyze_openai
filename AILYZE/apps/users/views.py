from django.shortcuts import render,HttpResponse,redirect
from apps.users.forms import UserChangePassword,RegisterUser,SummerizeType,SPecificQuestion,ThemeType,IdentifyViewpoint,CompareViewpoint,UplaodFileForm,ContactForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.views import LogoutView
from django.urls import reverse_lazy
from django.views import View
from apps.users.models import User,UserQuery,Files
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth import update_session_auth_hash
from apps.users.enum import Anaylsis
from django.views.generic.list import ListView
from apps.users.utils import FileHandler,SumarrizeClass,QuestionClass,ThemeAnalysisClass,FrequencyHandlerClass,CompareViewPointsClass





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
    





class UploadFileChoice(View):
                
    def get(self, request):
        form = UplaodFileForm()
        context = {'form': form}
        return render(request, "filedata.html", context)

    def post(self, request):
        user = self.request.user
        form = UplaodFileForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            obj.email = user.email
            obj.save()       
            return redirect('/get-form')
        return render(request, "filedata.html", {'form': form
                   })
    

class Getchoices(View):
    def get(self, request):
        choices = Anaylsis.choices()
        context = {'choices': choices}
        return render(request, "choices.html", context)




class UserQuestion(View):
    a = {
        Anaylsis.Summarize.value: lambda request: render(request,'chioceform.html',{'forms':SummerizeType()}),
        Anaylsis.Ask_a_specific_question.value:  lambda request:render(request,'chioceform.html',{'forms':SPecificQuestion()}),
        Anaylsis.Conduct_thematic_analysis.value: lambda request: render(request,'chioceform.html',{'forms':ThemeType()}),
        Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:  lambda request: render(request,'chioceform.html',{'forms':IdentifyViewpoint()}),
        Anaylsis.Compare_viewpoints_across_documents.value:  lambda request: render(request,'chioceform.html',{'forms':CompareViewpoint()})
    }
    def post(self, request):
        choice = request.POST.get('choice')
        data=request.session['choice']=choice
        render_fun = self.a.get(choice)
        if not render_fun:
            return render(request,'chioceform.html',{'forms':SummerizeType()}),
        return render_fun(request)





class ProcessQuery(View):
    def post(self, request):
        form = UplaodFileForm(request.POST, request.FILES)
        uploaded_file = request.FILES.getlist('file')
        upload_option = request.POST['upload_option']
        choice = request.session.get('choice')
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
                    if request.user.is_autheticate:
                        UserQuery.objects.create(user=self.request.user,question={'Summary_type':summary_type,"Summary_instruction":summary_instruction},answer=response['summary'])
                        return redirect('/user-profile')
                    else:
                        pass
                
                return render(request,'filedata.html',{'forms':forms})
            if choice==Anaylsis.Ask_a_specific_question.value:
                forms=SPecificQuestion(request.POST)
                if forms.is_valid():
                    question=forms.cleaned_data['question']
                    quesion_instruction=forms.cleaned_data['instruction']
                    question_keyword=forms.cleaned_data['keywords']
                    instance=QuestionClass(question=question,keywords=question_keyword,instruction=quesion_instruction)
                    response=instance.answer_question(df)
                    
                    UserQuery.objects.create(user=self.request.user,question={'question':question,"quesion_instruction":quesion_instruction,'question_keyword':question_keyword},answer=response)
                    return redirect('/user-profile')
                return render(request,'filedata.html',{'forms':forms})
            if choice==Anaylsis.Conduct_thematic_analysis.value:
                forms=ThemeType(request.POST)
                if forms.is_valid():
                    theme_type=forms.cleaned_data['theme_type']
                    instruction=forms.cleaned_data['instruction']
                    instance=ThemeAnalysisClass(theme_type=theme_type,theme_instruction=instruction)
                    response=instance.check_theme_type(df)
                    UserQuery.objects.create(user=self.request.user,question={'theme_type':theme_type,"instruction":instruction},answer=response['answer'])
                    return redirect('/user-profile')
                return render(request,'filedata.html',{'forms':forms})
            
            if choice==Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:
                forms=IdentifyViewpoint(request.POST)
                if forms.is_valid():
                    instruction=forms.cleaned_data['instruction']
                    instance=FrequencyHandlerClass(frequency_viewpoint=instruction)
                    response=instance.call_frequency(df)
                    UserQuery.objects.create(user=self.request.user,question={" Frequency_instruction":instruction},answer=response['answer'])
                    return redirect('/user-profile')
                return render(request,'filedata.html',{'forms':forms})
                
            if choice==Anaylsis.Compare_viewpoints_across_documents.value:
                forms=CompareViewpoint(request.POST)
                if forms.is_valid():
                    question=forms.cleaned_data['question']
                    instruction=forms.cleaned_data['instruction']
                    instruction_only=forms.cleaned_data['instruction_only']
                    keywords=forms.cleaned_data['keywords']
                    instance=CompareViewPointsClass(user=self.request.user,question_compare_groups=question,instruction_compare_groups=instruction,keywords_only=keywords,instructions_only=instruction_only)
                    response=instance.answer_question(df)
                    UserQuery.objects.create(question={" question":question,'instruction':instruction,'keywords':keywords,'instruction_only':instruction_only},answer=response)
                    return redirect('/user-profile')
                return render(request,'filedata.html',{'forms':forms})
                     
        else:
            form = UplaodFileForm(request.POST, request.FILES)
        return render(request, 'filedata.html', {'form': form})




class TryTech(View):
    def get(self, request):
        form = UplaodFileForm()
        context = {'form': form}
        return render(request, "techdemo.html", context)
 


    
            

class ShowData(ListView):
    model=UserQuery
    template_name='userdata.html'
    context_object_name='userdata'

      
    def get_queryset(self):
        queryset = super().get_queryset()
        user = self.request.user
        queryset = queryset.filter(user=user)
        return queryset
  


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