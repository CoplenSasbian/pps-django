from django.urls import path,include,re_path
from model import model_ws_server
websocket_urlpatterns = [
    path('ws/model',model_ws_server.ModelServer),
]