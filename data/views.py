import base64
import json
from math import ceil
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from data.cache import deletePageCache, getPageFromCache
from model.models import Model
from card_model.v2_model import ModelPersistence,PredictModel
import data.models as  md
from model.cache import get_model_by_id
from data.models import getModelFormDataType

from django.core.serializers import serialize
from django.views.decorators.http import require_http_methods, require_GET, require_POST
# Create your views here.
import pandas as pd
import io
def _auto_cast(v:str,type:str):
    if not v:
        return v
    if type=='string':
        return v
    elif type == 'integer':
        return int(v)
    elif type == 'floating':
        return float(v)
    else :
        raise Exception('not suppot yet')
    
def _fill_Model(mo:ModelPersistence,model,d:dict):
    dataType = mo.getDataType()
    ylable = mo.getYLable()
    mo = model()
    for i,v in dataType.items():
        if i == ylable:
            continue
        if not i in d:
                raise Exception (i + " can't be None")
        setattr(mo,i,_auto_cast(d.get(i),v)) 
    return mo



@require_POST
def getPage(request:HttpRequest):
    data:dict = json.loads(request.body.decode())
    modelId = data['modelId']
    pageSize = data['pageSize']
    page = data['pagenum']
    orederCol = None
    orederType = None
    if 'col' in data:
        orederCol = data['col']
    if 'type' in data:
        orederType = data['type']
    
    model =  ModelPersistence (get_model_by_id(int(modelId)))
    dataType = model.getDataType()
    dataModel = md.getModelFormDataType(model)
    
    optChain = dataModel.objects;  
    Model.objects.filter()
    
    if 'filter' in data:
        filter = data['filter']
        filterDict = dict()
        for item in filter:
            k = item['col']
            v = item['value']
            if v == '':
                continue
            if dataType.get(k) == 'string':
                filterDict[k+'__contains'] = v
            else:
                filterDict[k] = v
        optChain = optChain.filter(**filterDict)
        
    
    if orederCol and orederType:
        if orederType != 'ascend':
            orederCol = '-' + orederCol
        
    optChain = optChain.all()
    
    
    
    if orederCol:
        optChain = optChain.order_by(orederCol)
    
    key = [optChain,pageSize,page]
    data,count =  getPageFromCache(key)
    
    data = serialize("json", data)
    data = '{{"count":{},"data":{}}}'.format(count,data)
    return HttpResponse(data)

@require_POST
def readToJSON(request:HttpRequest):
    data = json.loads(request.body.decode())
    file = data['data']
    type = data['dataType']
    bytes= base64.b64decode(file.split(',')[1])
    df = None
    if type == 'csv':
        df = pd.read_csv(io.BytesIO(bytes))
    else:
        df = pd.read_excel(io.BytesIO(bytes))
    return HttpResponse(df.to_json())

@require_POST
def addDatas(requset:HttpRequest):
    try:
        data:dict = json.loads(requset.body.decode())
        model = ModelPersistence( get_model_by_id(data['modelId']))
        modelType = getModelFormDataType(model)
        datas: list =  data['data']
        modelList = []
        for i in  datas:
            mo = _fill_Model(model,modelType,i)
            modelList.append(mo)
        modelType.objects.bulk_create(modelList)
        deletePageCache()
        return HttpResponse('Success')
    except Exception as e:
        return HttpResponse(str(e))
    
@require_POST
def pridictData(request:HttpRequest):
    data:dict = json.loads(request.body.decode())
    model = PredictModel( get_model_by_id(data['modelId']))
    pdData = pd.DataFrame(data['data'])
    woettable,result = model.predict(pdData)
    py  = [v for i,v in result ]
    result = {
        'woeTable':woettable,
        'result':py
    }
    return HttpResponse(json.dumps(result))
    
@require_GET
def getCoefficient (request: HttpRequest):
    id = int(request.GET.get('modelId'))
    model = get_model_by_id(id)
    model = ModelPersistence(model)
    return HttpResponse(json.dumps(model.getLRModel().coef_.tolist()))


@require_GET
def getIntercept (request: HttpRequest):
    id = int(request.GET.get('modelId'))
    model = get_model_by_id(id)
    model = ModelPersistence(model)
    return HttpResponse(json.dumps(model.getLRModel().intercept_.tolist()))