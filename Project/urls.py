from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from healthapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication
    path('login/', auth_views.LoginView.as_view(template_name='auth/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', views.register, name='register'),
    
    # Main app URLs
    path('', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('chat/', views.chat, name='chat'),
    path('api/chat/', views.chat, name='chat_api'),
    
    # Dashboard functionality
    path('add_meal/', views.add_meal, name='add_meal'),
    path('complete_exercise/', views.complete_exercise, name='complete_exercise'),
    path('update-water/', views.update_water, name='update_water'),
] 