from django.urls import path
from .  import views

urlpatterns = [
    path('to', views.home, name='home'),
]