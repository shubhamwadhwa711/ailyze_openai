from apps.users import views
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',views.Home.as_view(),name='home'),
    path('register/',views.Register.as_view(), name="register"),
    path('login/',views.LoginView.as_view(), name="login"),
    path('logout/',views.LogoutView.as_view(), name="logout"),
    path("change-password/",views.ChangePassword.as_view(), name="change-password"),
    path('user-profile/', views.UserProfile.as_view(),name="user-profile"),
    path('contact/', views.Contactform.as_view(),name="contact"),
    path('about/', views.About.as_view(), name='about'),
    path('get-form/',views.Getchoices.as_view(),name='get-form'),
    path('query/',views.UserQuestion.as_view(),name='query'),
    path('processing/',views.ProcessQuery.as_view(),name='processing'),
    path('view-list/', views.ShowData.as_view(), name='view-list'),
    path('detailpage/<int:pk>', views.DetailPage.as_view(),name="detailpage"),
    path('old-files/', views.oldFiles.as_view(), name="old-files")
  
  
    
]+static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)+static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)

    