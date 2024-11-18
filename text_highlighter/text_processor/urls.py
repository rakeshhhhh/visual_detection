from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate_pdf/', views.generate_pdf, name='generate_pdf'),
]
