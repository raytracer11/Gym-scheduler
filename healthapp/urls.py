from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('add-meal/', views.add_meal, name='add_meal'),
    path('complete-exercise/', views.complete_exercise, name='complete_exercise'),
    path('chat/', views.chat, name='chat'),
    path('update-water/', views.update_water, name='update_water'),
] 