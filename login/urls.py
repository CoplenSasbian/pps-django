from django.contrib import admin
from django.urls import path
from . import views

urlpatterns =[
    path('login', views.log),
    path('has',views.has),
    path('has_email',views.has_emial),
    path('registered',views.reg),
    path('refine',views.refine),
    path('current',views.current),
    path('hm',views.hw),
    path('out',views.logout_),
    path('getinfo',views.getUserInfo),
]