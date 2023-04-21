import json
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.views.decorators.http import require_http_methods, require_GET, require_POST
import pytz
import requests
from django.core import serializers
from login.cache import del_cache_by_id, get_user_by_id


# Create your views here.
def responeSucceed(message="succeeded"):
    return HttpResponse(json.dumps({"state": True, "message": message}))


def responeFailed(message: str = "failed"):
    return HttpResponse(json.dumps({"state": False, "message": message}))




@require_POST
def has(request: HttpRequest):
    """
    是否存在该用户
    POST username
    """
    username = json.loads(request.body.decode())["username"]
    exists = User.objects.filter(username=username).exists()
    if exists:
        return responeSucceed("has user")
    else:
        return responeFailed("could no find user")


@require_POST
def log(request: HttpRequest):
    """
    登录
    """
    body = request.body.decode()
    obj = json.loads(body)
    username = obj["username"]
    password = obj["password"]

    logUser = authenticate(username=username, password=password)
    if logUser:
        login(request, logUser)
        return responeSucceed("center")
    else:
        return responeFailed()


@require_GET
def has_emial(request: HttpRequest):
    """
    是否有该邮箱
    """
    email = request.GET.get("email")
    exists = User.objects.filter(email=email).exists()
    if exists:
        return responeSucceed()
    else:
        return responeFailed()


@require_POST
def reg(request: HttpRequest):
    """
    注册
    """
    body = request.body.decode()
    obj = json.loads(body)
    username = obj["username"]
    password = obj["password"]
    email = obj["email"]

    newUser = User.objects.create_user(
        username=username, password=password, email=email
    )
    newUser.save()
    if newUser:
        login(request, newUser)
        return responeSucceed("center")
    else:
        return responeFailed()


@require_POST
def refine(request: HttpRequest):
    """
    完善姓名
    """
    body = request.body.decode()
    obj = json.loads(body)
    username = obj["username"]
    firstname = obj["firstname"]
    lastname = obj["lastname"]

    if request.user.username != username:
        return responeFailed("当前用户未登录！")

    user = User.objects.get(username=username)
    if not user:
        return responeFailed("当前用户未不存在！")
    user.first_name = firstname
    user.last_name = lastname
    user.save()
    del_cache_by_id(user.pk)
    return responeSucceed("center")

beijing_tz = pytz.timezone('Asia/Shanghai')
@require_GET
def current(request: HttpRequest):
    id = request.user.id
    user = None
    if id:
       user = get_user_by_id(id)
    if user:
        info = {
            "username": user.get_username(),
            "firstname":user.first_name,
             "lastname":user.last_name,
            "fullname": user.get_full_name(),
            "email": user.email,
            "last_login": user.last_login.astimezone(beijing_tz).strftime('%Y-%m-%dT%H:%M:%S.%f'),
            "join_date":user.date_joined.astimezone(beijing_tz).strftime('%Y-%m-%dT%H:%M:%S.%f'),
            "is_superuser": user.is_superuser,
        }
        return responeSucceed(info)
    return responeFailed(None)


@require_GET
def hw(request: HttpRequest):
    user = get_user_by_id(1)
    login(request, user)
    return responeSucceed()


@require_GET
def logout_(request: HttpRequest):
    logout(request)
    request.session.clear()
    return responeSucceed()


@require_GET
def getUserInfo(request: HttpRequest):
    id = request.GET.get("id")
    user = get_user_by_id(id)
    if user:
        info = {
            "username": user.get_username(),
            "fullname": user.get_full_name(),
            "email": user.email,
            "last_login": user.last_login.timestamp(),
            "is_superuser": user.is_superuser,
        }
        return responeSucceed(info)
    return responeFailed()


@require_POST

def updateCurrent(request: HttpRequest):
    raw = request.body.decode()
    user = json.loads(raw)
    
    if user['username'] == request.user.username:
        dbUser = User.objects.get(username= user['username'])
        dbUser.email = user['email']
        dbUser.first_name = user['firstname']
        dbUser.last_name = user['lastname']
        dbUser.save()
        del_cache_by_id(dbUser.pk)
        return HttpResponse("",status = 204)
    return HttpResponse("",status = 401)