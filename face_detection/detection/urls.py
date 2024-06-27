from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('video_feed/', views.video_feed, name='video_feed'),
]