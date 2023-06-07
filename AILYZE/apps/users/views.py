from django.shortcuts import render,HttpResponse,redirect
from apps.users.forms import UserChangePassword,RegisterUser,SummerizeType,SPecificQuestion,ThemeType,IdentifyViewpoint,CompareViewpoint,UplaodFileForm
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
from apps.users.utils import FileHandler,SumarrizeClass



# Create your views here.


class Home(View):
    def get(self,request):
        print("---")
        
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
    def handle_uploaded_file(f):  
        with open('myapp/static/upload/'+f.name, 'wb+') as destination:  
            for chunk in f.chunks():  
                destination.write(chunk)  
                
    def get(self, request):
        form = UplaodFileForm()
        context = {'form': form
                   }
        return render(request, "filedata.html", context)

    def post(self, request):
        user = self.request.user
        form = UplaodFileForm(request.POST, request.FILES)
        try: 
            uploaded_file = request.FILES.getlist('file')
            if not isinstance(uploaded_file, list):
                uploaded_file = [uploaded_file]
            instance=FileHandler(uploaded_file)
            df=instance.upload_documents(max_documents=50,max_words=10000)
            print(" succesfully", df)
        except Exception as e:
                print("This is is Exception as ", e)
        if form.is_valid():
            obj = form.save()
            obj.email = user.email
            # obj.save()
           
            return redirect('/get-form')
        return render(request, "filedata.html", {'form': form
                   })
    




class Getchoices(View):
    def get(self,request):
        choices=Anaylsis.choices()
        context={
            'choices':choices,
        }
        return render(request,"choices.html",context)
    


class UserQuery(View):
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
        render_fun = self.a.get(data)
        if not render_fun:
            return render(request,'chioceform.html',{'forms':SummerizeType()}),
        return render_fun(request)
    


class ProcessQuery(View):
    # a = {
    #     Anaylsis.Summarize.value: lambda request: render(request,'chioceform.html',{'forms':SummerizeType()}),
    #     Anaylsis.Ask_a_specific_question.value:  lambda request:render(request,'chioceform.html',{'forms':SPecificQuestion()}),
    #     Anaylsis.Conduct_thematic_analysis.value: lambda request: render(request,'chioceform.html',{'forms':ThemeType()}),
    #     Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:  lambda request: render(request,'chioceform.html',{'forms':IdentifyViewpoint()}),
    #     Anaylsis.Compare_viewpoints_across_documents.value:  lambda request: render(request,'chioceform.html',{'forms':CompareViewpoint()})
    # }
    def post(self,request):
        choice=request.session.get('choice')
        variables = {
        "Summarize": ["summary", "instruction"],
        "Ask_a_specific_question": ["question", "instruction", "keywords"],
        "Conduct_thematic_analysis": ["theme_type", "instruction"],
        "Identidy_which_document_contain_a_certain_viewpoint": ["instruction"],
        "Compare_viewpoints_across_documents": ["instruction", "question"]
    }    
        if choice in variables:
            print("-------------------------------------", choice)
            values = [request.POST.get(var, '') for var in variables[choice]]
            summary_value = values[variables[choice].index("summary")]
            instruction_value = values[variables[choice].index("instruction")]
            quetion = summary_value + instruction_value
            answer = "good"
            UserQuery.objects.create(question=quetion, answer=answer , user = self.request.user)
            return HttpResponse("Done ")
        return HttpResponse("Not Done ")

    


    
