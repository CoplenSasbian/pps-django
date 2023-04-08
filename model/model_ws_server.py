import base64
import json
from channels.generic.websocket import WebsocketConsumer
from django.contrib.sessions.backends.db import SessionStore
from  django.contrib.auth.models import User
from enum import Enum
from card_model.v2_model import TrainingModel
import io
import pandas as pd
import data.models as DataModel
from django.core.cache import cache

from model.models import Model
class ModelServer(WebsocketConsumer):
    
    def connect(self):
        self.accept()
        
        
    def disconnect(self, close_code):
        self.session.delete('currenntModel')
        
    def setData(self,data:dict):
        sp:str = data['data']
        inor:bool  = data['ignoreIndex']
        bytes= base64.b64decode(sp.split(',')[1])
        type = data['fileType']
        if type == 'csv':
            file =  pd.read_csv(io.BytesIO(bytes))
        else:
            file =  pd.read_excel(io.BytesIO(bytes))
        model = TrainingModel()
        model.loadData(file,inor)
        self.session.setdefault('currenntModel',model)
        return self.Succeed('')
        
    def getDescript(self):
        model = self.getCurrentUserModel()
        jsonstr = model.getDescriptor().to_dict()
        return self.Succeed(jsonstr)
    
    def setMissingMethod(self,obj:dict):
        model = self.getCurrentUserModel()
        model.setMissingMethod(obj['lable'],obj['method'],obj['default'])
        return self.Succeed('')
    
    def getDataType(self):
        model = self.getCurrentUserModel()
        res = model.getDataType()
        return self.Succeed(res)
        
    def getMissingInfo(self):
        model = self.getCurrentUserModel()
        res = model.getMissingInfo()
        return self.Succeed(res)
        
    def setYlable(self,ylable:str):
        model = self.getCurrentUserModel()
        model.setYLable(ylable)
        model.clu_odds()
        return self.Succeed('')
         
    def getBinner(self):
        model = self.getCurrentUserModel()
        res = model.getWoeIvJson()
        return self.Succeed(res)
        
    def useLable(self,data:dict):
        model = self.getCurrentUserModel()
        model.setLableUse(data['lable'],data['use'])
        return self.Succeed('')
        
    def getLableUsed(self):
        model = self.getCurrentUserModel()
        return self.Succeed(model.getLableUse())
    
    def gennerateModel(self,data:dict):
        model = self.getCurrentUserModel()
        model.setBenchmarkScore(data['base'])
        model.setPDOScore(data['pdo'])
        model.setOdds(data['odds'])
        model.genLinnerRegressionModel()
        return self.Succeed('')

    def getCurrentUserModel(self):
        model:TrainingModel = self.session.get('currenntModel')
        return model
    
    def testModel(self):
        model = self.getCurrentUserModel()
        result = model.testModel()
        return self.Succeed(result)
    
    def saveModel(self,data:dict):
        model = self.getCurrentUserModel()
        model.setName(data['name'])
        model.setUser(self.session.get('_auth_user_id'))
        DataModel.createModelFormDataType(model)
        model.save()
        cache.delete('model.all')
        from data.admin import registed
        registed(Model.objects.all())
        return self.Succeed('')
    def getInterceptCoefficient(self):
        model = self.getCurrentUserModel()
        model = model.getLRModel()
        return  self.Succeed({'coefficient':model.coef_.tolist(),'intercept':model.intercept_.tolist()})
    
    def getOdds(self):
        model = self.getCurrentUserModel()
        return  self.Succeed(model.getOdds())
    
    def Succeed(self,data):
        return{
            'type':'',
            'state':True,
            'data':data
        }
        
    def Failed(self,message):
        return  {
            'type':'',
            'state':False,
            'data':message
        }

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        if text_data_json['type']=='cookie':
            self.session = SessionStore(text_data_json['data'])
        else:
            fun = getattr(self,text_data_json['type'])
            if fun:
                if text_data_json['data']:
                    ret = fun(text_data_json['data'])
                else:
                    ret = fun()
                ret['type'] = text_data_json['type']
                self.send(json.dumps(ret))
            
            

