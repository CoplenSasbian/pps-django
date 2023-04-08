import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model.models import Model
from datetime import datetime
from django.contrib.auth.models import User
import json
from sklearn.metrics import roc_curve, auc
import pickle
class ModelPersistence():
    
    def __init__(self,model:Model = None):
        if model :
            self.model = model
            self.raDdescriptor = pd.read_json(model.descriptor)
            self.rawDataType = json.loads(model.data_type)
            self.rawLableUse = json.loads(model.lable_use)
            self.rawMissingInfo = json.loads(model.missing_info)
            self.rawWoeIvTable = self.load_weo_iv_table()
            self.rawLRModel = pickle.loads(self.model.lr_model_dump)
            self.rawRocData = json.loads(model.rocData)
        else:
            self.model = Model()
    
    def setName(self,name:str):
        self.model.name = name
    
    def setDescriptor(self,desc:pd.DataFrame):
        self.raDdescriptor = desc
    
    def setCreateDate(self,date:datetime = datetime.now()):
        self.model.create_date = date
        
    def setUser(self,id:int):
        self.model.user = User.objects.get(id = id)
      
    def setYLable(self,yLable:str):
        self.model.ylable = yLable
    
    def setWoeIvTable(self,data:dict):
        self.rawWoeIvTable = data
    
    def setDataType(self,data:dict):
        self.rawDataType = data
        
    def setMissingInfo(self,data:dict):
        self.rawMissingInfo = data
        
    def setLRModel(self,data:LogisticRegression):
        self.rawLRModel = data
        
    def setBenchmarkScore(self,n:int):
        self.model.base_score = n
        
    def setPDOScore(self,n:int):
        self.model.pdo_score = n
    
    def setLableUseInfo(self,data:dict):
        self.rawLableUse = data
    
    def setRocData(self,data):
        self.rawRocData = data
        
    def getRocData(self):
        return self.rawRocData
    
    def getMissingInfo(self):
        return self.rawMissingInfo
    
    def getDataType(self):
        return self.rawDataType
    
    def getWoeIvTable(self):
        return self.rawWoeIvTable
    
    def getLableUse(self):
        return self.rawLableUse
    
    def getDescriptor(self):
        return self.raDdescriptor
    
    def getLRModel(self):
        return self.rawLRModel
    
    def getYLable(self):
        return self.model.ylable
    
    def getUsedLables(self):
        '''python 3.7 或 cpython 3.6 以上。其余不保持插入顺序 '''
        ylable = self.getYLable()
        return [k for k,v in self.getLableUse().items() if v and k != ylable]
    
    def getName(self):
        return self.model.name
    
    def getOdds(self):
        return self.model.odds
    
    def setOdds(self,odds):
        self.model.odds = odds
    
    def serialize_woe_iv_table(self):
        nd = dict()
        for i,v in self.rawWoeIvTable.items():
            nd[i] = v.to_json()
        return json.dumps(nd)
        
    def load_weo_iv_table(self):
        di:dict = json.loads(self.model.woe_iv_table)
        nd = dict()
        for i,v in di.items():
            nd[i] = pd.read_json(v)
        return nd        

    def save(self):
        self.model.woe_iv_table = self.serialize_woe_iv_table()
        self.model.data_type = json.dumps(self.rawDataType)
        self.model.missing_info = json.dumps(self.rawMissingInfo)
        self.model.lr_model_dump =  pickle.dumps(self.rawLRModel)
        self.model.lable_use = json.dumps(self.rawLableUse)
        self.model.descriptor = self.raDdescriptor.to_json()
        self.model.rocData = json.dumps(self.rawRocData)
        if not self.model.create_date :
            self.model.create_date = datetime.now()
        self.model.save()
    
    
class PredictModel(ModelPersistence):
    def __init__(self, model: Model = None):
        super().__init__(model)
    
    
    def _fill_missing_values(self,missingLable:str,data:pd.DataFrame):
        '''
            随机森林 补充缺失值
            @pram:
                missingLable :缺失值的列
                yLble: 弃用
            @return
                DataFrame (该列补充的全部数据)
        '''
        data = data.copy()
        le = LabelEncoder()

        #字符数值化
        for i in data.head():
            if not pd.api.types.is_numeric_dtype( data[i]):
                data[i] = le.fit_transform(data[i])

        #分开缺失值
        known = data[data[missingLable].notnull()]
        unknown = data[data[missingLable].isnull()]

        x_train = known.drop(columns=missingLable)
        y_train = known[missingLable]
        x_test = unknown.drop(columns=missingLable)
        rfr = RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
        rfr.fit(x_train,y_train) 
        pred_y  = rfr.predict(x_test)
        data.loc[data[missingLable].isna(), missingLable] = pred_y
        
        return data[missingLable]
        
        
    def setMissingMethod(self,lablename:str,method:str,default:str):
        '''
            设置缺失填充信息
            @parm
                lablename:要设置的列明
                method:处理方法  （自成分箱 预测填充 中位数 众数 自定义）
                default:处理方法为自定义时的天填充值
        '''
        missing = self.getMissingInfo()
        dataType = self.getDataType()
        if method == '自定义':
            dt = dataType[lablename]
            if pd.api.types.is_bool(dt):
                missing[lablename]["default"] = bool(default)
            elif pd.api.types.is_float_dtype(dt):
                missing[lablename]["default"] = float(default)
            elif pd.api.types.is_integer(dt):
                missing[lablename]["default"] = int(dataType[lablename])
            elif pd.api.types.is_string_dtype(dt):
                missing[lablename]["default"] = default
        missing[lablename]["missingMethod"] = method
        
    def _fix_missing_value_and_gen_data(self,data:pd.DataFrame,lableName:str):
        '''
            通过缺失信息表( getMissingInfo 可获得获得)填充缺失信息
        '''
        ylable = self.getYLable()
        missing = self.getMissingInfo()
        needMissing = all( [missing[lableName]["missingMethod"] == "自成分箱" ,  missing[lableName]['nullCount'] != 0])
        data = pd.DataFrame({'X':data[lableName],'Y':data[ylable]})
        missingMethod = missing[lableName]["missingMethod"]
        if missingMethod == '预测填充':
            data['X'] = self._fill_missing_values('X',data)
        elif  missingMethod == '众数':
            most = data['X'].notnull().mode()
            data['X'] = data['X'].fillna(most)
        elif missingMethod == '中位数':
            mid = data['X'].notnull().median()
            data['X'] = data['X'].fillna(mid)
        elif missingMethod == '自定义':
            data['X'] = data['X'].fillna(self.missing[lableName]['default'])
        return data,needMissing
    
    def setLableUse(self,lable,u):
        '''
            设置列是否使用
            @pram
                lable:列名
                u: True or False
        '''
        self.rawLableUse[lable] = u
    
    def woeTranformFroTrainning(self,data:pd.DataFrame,woeIv:dict):
        '''
        将data对应lable替换成对应的Woe 由于时训练集 需要进行空值处理
        '''
        lableUse = self.getLableUse()
        missingMethod = self.getMissingInfo()
        for col in data.columns:
            if not lableUse[col]:
                continue
            ret,_ =self._fix_missing_value_and_gen_data(data,col)
            data[col] = ret['X']
            if col in woeIv:
                bins = woeIv[col]['bin']
                woed = woeIv[col]['woe']
                intervals = [b for b in bins]
                for i, interval in enumerate(intervals):
                    print(interval)
                    if pd.api.types.is_interval(interval):
                       
                        #区间的分箱
                        data.loc[(data[col] > interval.left) & (data[col] <= interval.right),col] = woed[i]
                        
                    else:
                        #值的分箱
                        data.loc[data[col]==(interval),col]= woeIv[col]['woe'][i]
                       
                if missingMethod[col]['missingMethod']=='自成分箱': #空值处理 
                    laDf:pd.DataFrame = woeIv[col]
                    res = laDf.loc[laDf['bin']=='Missing','woe']
                    if not res.empty:
                        res = res.item()
                        data[col] = data[col].fillna(res)
        return data
    
    def woeTranformForPridict(self,data:pd.DataFrame,woeIv:dict):
        '''
        将data对应lable替换成对应的Woe 预测不处理空值
        '''
        lableUse = self.getLableUse()
        missingMethod = self.getMissingInfo()
        for col in data.columns:
            if not lableUse[col]:
                continue
            if col in woeIv:
                bins = woeIv[col]['bin']
                woed = woeIv[col]['woe']
                intervals = [b for b in bins]
                for i, interval in enumerate(intervals):
                    if pd.api.types.is_dict_like(interval):
                        #区间的分箱
                        data.loc[(data[col] > interval['left']) & (data[col] <= interval['right']),col] = woed[i]
                        
                    else:
                        #值的分箱
                        data.loc[data[col]==(interval),col]= woeIv[col]['woe'][i]
                       
                if missingMethod[col]['missingMethod']=='自成分箱': #空值处理 
                    laDf:pd.DataFrame = woeIv[col]
                    res = laDf.loc[laDf['bin']=='Missing','woe']
                    if not res.empty:
                        res = res.item()
                        data[col] = data[col].fillna(res)
        return data    
    
    def predict(self,data:pd.DataFrame):
        data = self.woeTranformForPridict(data,self.getWoeIvTable())
        useLables = self.getUsedLables()
        xdata = data[useLables]
        lrmode = self.getLRModel()
        return data.to_dict(),lrmode.predict_proba(xdata)
     
        
        
        
class TrainingModel(PredictModel):
    def __init__(self) -> None:
        super().__init__()
        
    def loadData(self,data:pd.DataFrame,ignoreIndex:bool):
        if ignoreIndex:
            data = data.drop(data.columns[0], axis=1)
        else:
            data = data
        self.data = data
        self.traindata ,self.testDate=  train_test_split(data,test_size=0.25,random_state=24)
    
        self.__gen_missing_info()
        self.__gen_types()
        self.__gen_descript()
        self.__gen_lableUse()
    def clu_odds(self):
        result = self.data[self.getYLable()]
        _sum = result.sum()
        self.setOdds(_sum/(result.count()-_sum))
        
    def __gen_missing_info(self):
        '''
        @return 
            DataFrame(
                {
                'nullCount':空值数
                'total':总个数
                'missingMethod':缺失处理方法 （自成分箱 或 预测补充)
                }
            )
        '''
        missingD = pd.DataFrame()
        for name in self.data.keys():
            nullCount = self.data[name].isnull().sum()
            count = nullCount + self.data[name].notnull().sum()
            missingD[name] = {"nullCount": int(nullCount),"total":int(count),"missingMethod":"自成分箱","default":None}
        self.setMissingInfo(missingD.to_dict())
    def __get_lables(self):
        return self.data.head()
    
    def __gen_types(self):
        itypes = dict()
        for i in self.__get_lables():
           itypes[i] = pd.api.types.infer_dtype(self.data[i])
        self.setDataType(itypes)
        
    def __gen_descript(self):
        self.setDescriptor(self.data.describe())
        
    def __gen_lableUse(self):
        useLables = dict()
        for i in self.data.keys():
            if i != self.getYLable():
                useLables[i] = True
        self.setLableUseInfo(useLables) 
    
    def __binning_continue_value(self,lableName:str):
        '''
            qcat 分箱 计算WOE IV
            parm:
                @lableName   分箱列
                Y结果列用 set_y_lable 设置
            return:
                 DataFrame({
                     'bin': 分箱箱名（ 范围(开始,结束] 或者缺失值Missing)
                     'woe': WOE值
                     'iv’ : 每个分箱的iv值
                 })
        '''
       
        data = self.traindata.copy()
        data,needMissing = self._fix_missing_value_and_gen_data(data,lableName)
            
        X = data['X']
        Y = data['Y']
        n = 20
        r = 0
        
        good = Y.sum()
        bad = Y.count() - good
       
        
        while np.abs(r) < 0.99:
            cd = pd.qcut(X, n, duplicates="drop")
            if needMissing:
                cd = cd.cat.add_categories(['Missing'])
                cd = cd.fillna('Missing')          
            
            temp_data = pd.DataFrame({'X':X,'Y':Y,"bin":cd})
            group_data = temp_data.groupby(["bin"])
            r,_ = stats.spearmanr(group_data['X'].mean(),group_data['Y'].mean())
            n = n -1
        
        result = pd.DataFrame()
        result['good'] = group_data.sum()["Y"]
        result['count'] = group_data.count()["Y"]
        result['bad'] =  result['count'] - result['good']
        result['woe'] =  np.log((result['bad']/bad ) / (result['good']/good))
        result['iv'] = (result['bad']/bad - result['good']/good)*result['woe']
        
        
        di = pd.DataFrame()
        di['bin'] = group_data.groups.keys()
        di['woe'] = result['woe'].values
        di['iv'] = result['iv'].values
        return di
        
    def __biinning_category_value(self,lableName:str):
        '''
            离散值个离散变量成一个分箱,计算WOE IV
            pram:
                @lableName:进行分箱的列
            return:
            DataFrame({
                'bin': 分箱箱名（ 范围(开始,结束] 或者缺失值Missing)
                'woe': WOE值
                'iv’ : 每个分箱的iv值
            })
        '''
        data = self.traindata.copy()
        data, _ = self._fix_missing_value_and_gen_data(data,lableName)
        good = data['Y'].sum()
        bad = data['Y'].count() - good
        groupData = data.groupby('X',as_index=True)
        re = pd.DataFrame()
        re['bin'] = groupData.groups.keys()
        tmp = groupData['Y']
        tmp = tmp.sum()
        re['good'] = tmp.values
        re['bad'] = groupData['Y'].count().values - re['good']
        re['woe'] = np.log((re['bad']/bad)/(re['good']/good))
        re['iv'] =( re['bad'] / bad - re['good'] / good ) * re['woe']
        res = pd.DataFrame()
        res['bin'] = re['bin']
        res['woe'] = re['woe']
        res['iv'] = re['iv']
        return res
    
    def __gen_woe_iv_table(self):
        '''
            建立每一列分箱 WOE IV 表
            @return
                ditc{
                    '列名'-> dataframe{
                        bin:分箱
                        woe
                        iv
                    }
                } 
        '''
        result = dict()
        for i in self.data.keys():
            if i != self.getYLable():
                if pd.api.types.is_numeric_dtype( self.data[i]):
                   result[i] = self.__binning_continue_value(i)
                else:
                   result[i] = self.__biinning_category_value(i)
        return result
    
    def getPreNorWoeIv(self):
        '''
           获取每一列分箱 WOE IV 表(未经归一化)
            @return
                ditc{
                    '列名'-> dataframe{
                        bin:分箱
                        woe
                        iv
                    }
                } 
        '''
        return self.__gen_woe_iv_table()
    
    
    def getWoeIvJson(self):
        '''
           获取每一列分箱 WOE IV 表 json字符串(未经归一化)
        '''
        jsons = "{"
        for i,v in self.__gen_woe_iv_table().items():
            jsons+="\""
            jsons+=i
            jsons+="\":"
            jsons+=v.to_json()
            jsons+=","
        jsons = jsons[:-1]
        jsons+="}"
        return jsons
    
    def __gen_normalized_woe_iv_table(self):
        '''
             建立每一列分箱 WOE IV 表（归一化）
        '''
        woeIv = self.__gen_woe_iv_table()
        for i,v in woeIv.items():
            maxWoe = v['woe'].max()
            minWoe = v['woe'].min()
            v['woe'] = (v['woe'] - minWoe) / (maxWoe - minWoe)
        return woeIv
    
    def __genWoeIv(self):
        table = self.__gen_normalized_woe_iv_table()
        self.setWoeIvTable(table)
    
    def genLinnerRegressionModel(self):
        '''
            建立模型
        '''
        
        ylable = self.getYLable()
        self. __genWoeIv()
        useTable = self.getUsedLables()
        woe = self.getWoeIvTable()
        data = self.woeTranformFroTrainning(self.traindata.copy(),woe)
        x_train = data.loc[:,useTable]
        y_train = data[ylable]
        lr = LogisticRegression()
        lr.fit(x_train,y_train)
        self.setLRModel(lr)
    
    def testModel(self):
        ylable = self.getYLable()
        data = self.testDate
        data = self.woeTranformFroTrainning(data,self.getWoeIvTable())
        useLables = self.getUsedLables()
        xdata = data[useLables]
        lrmode = self.getLRModel()
        predictY =   lrmode.predict_proba(xdata)
        Y = data[ylable]
        predictY = [v for i,v in predictY ]
        fpr, tpr, thresholds = roc_curve(Y, predictY)
        roc_auc = auc(fpr,tpr)
        fpr = list(fpr)
        tpr = list(tpr)
        data = {'fpr':fpr,'tpr':tpr,'roc_auc':roc_auc}
        self.setRocData(data)
        return data
    
    
    