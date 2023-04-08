import json
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.models import User
from model.cache import get_all, get_model_by_id
from .models import Model

# Create your views here
from django.core import serializers

jsonFiled = (
    'name','descriptor','ylable','woe_iv_table','data_type','missing_info',
    'base_score','pdo_score','create_date','lable_use','user','rocData','odds'
)

jsonListFields=(
    'name',
)



def list(request: HttpRequest):
    all = get_all()
    data = serializers.serialize("json", all,fields=jsonListFields)
    return HttpResponse(data,content_type='application/json')


def detail(request: HttpRequest):
    id = int(request.GET.get('id'))
    data = get_model_by_id(id)
    data = serializers.serialize("json", [data],fields=jsonFiled)
    return HttpResponse(data,content_type='application/json')

