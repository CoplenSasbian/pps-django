import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split



'''
TODO :: 
1 将空值处理独立成一个方法 t
2 woe iv 值进行缓存 
3 处理不合理值的支持q
'''
class CardModelV1: 
    def __init__(self,data:pd.DataFrame) -> None:
        self.traindata ,self.testDate=  train_test_split(data,test_size=0.25,random_state=24)
        self.missing = pd.DataFrame()
        self.missingInfo()
        self.useLables = dict()
        
    
    def describe(self):
        return self.traindata.describe()
    
    def types(self):
        itypes = dict()
        for i in self.get_lables():
           itypes[i] = pd.api.types.infer_dtype(self.traindata[i])
        return itypes
    
    
    def missingInfo(self):
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
        if self.missing.empty:
            for name in self.traindata.keys():
                nullCount = self.traindata[name].isnull().sum()
                count = nullCount + self.traindata[name].notnull().sum()
                missingD[name] = {"nullCount": int(nullCount),"total":int(count),"missingMethod":"自成分箱","default":None}
            self.missing = missingD
            
        return self.missing
    
    def fill_missing_values(self,missingLable:str,data:pd.DataFrame):
        
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
        
        
               
    def set_missing_method(self,lablename:str,method:str,default:str):
        if method == '自定义':
            if pd.api.types.is_bool(self.traindata[lablename]):
                self.missing[lablename]["default"] = bool(default)
            elif pd.api.types.is_float_dtype(self.traindata[lablename]):
                self.missing[lablename]["default"] = float(default)
            elif pd.api.types.is_integer(self.traindata[lablename]):
                self.missing[lablename]["default"] = int(self.traindata[lablename])
            elif pd.api.types.is_string_dtype(self.traindata[lablename]):
                self.missing[lablename]["default"] = default
        self.missing[lablename]["missingMethod"] = method
        
    
    def set_y_lable(self, lableNmae:str):
        '''
            设置分箱时的 因变量列
        '''
        self.ylable = lableNmae
        
    
    def __fix_missing_value_and_gen_data(self,data:pd.DataFrame,lableName:str):
        
        needMissing = all( [self.missing[lableName]["missingMethod"] == "自成分箱" ,  self.missing[lableName]['nullCount'] != 0])
        data = pd.DataFrame({'X':data[lableName],'Y':data[self.ylable]})
        missingMethod = self.missing[lableName]["missingMethod"]
        if missingMethod == '预测填充':
            data['X'] = self.fill_missing_values('X',data)
        elif  missingMethod == '众数':
            most = data['X'].notnull().mode()
            data['X'] = data['X'].fillna(most)
        elif missingMethod == '中位数':
            mid = data['X'].notnull().median()
            data['X'] = data['X'].fillna(mid)
        elif missingMethod == '自定义':
            data['X'] = data['X'].fillna(self.missing[lableName]['default'])
        return data,needMissing
    
    def __binning_continue_value(self,lableName:str):
        '''
            qcat 分箱
            parm
                @lableName   分箱列
                Y结果列用 set_y_lable 设置
            return
                 DataFrame({
                     'bin': 分箱箱名（ 范围(开始,结束] 或者缺失值Missing)
                     'woe': WOE值
                     'iv’ : 每个分箱的iv值
                 })
        '''
       
        data = self.traindata.copy()
        data,needMissing = self.__fix_missing_value_and_gen_data(data,lableName)
            
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
        data = self.traindata.copy()
        
        data, _ = self.__fix_missing_value_and_gen_data(data,lableName)
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
        
    
    def get_woe_iv(self):
        result = dict()
        for i in self.traindata.head():
            if i != self.ylable:
                if pd.api.types.is_numeric_dtype( self.traindata[i]):
                   result[i] = self.__binning_continue_value(i)
                else:
                   result[i] = self.__biinning_category_value(i)
        return result
    
    def get_woe_iv_json(self):
        jsons = "{"
        for i,v in self.get_woe_iv().items():
            jsons+="\""
            jsons+=i
            jsons+="\":"
            jsons+=v.to_json()
            jsons+=","
        jsons = jsons[:-1]
        jsons+="}"
        return jsons
                            
    def get_lables(self):
        return  self.traindata.columns.to_list()
    
    def get_lable_use(self):
        if not self.useLables:
            for i in self.get_lables():
                self.useLables[i] = True
        return self.useLables       
    
    def set_lable_use(self,lable,u):
        self.useLables[lable] = u
    
    def get_normalized_woe(self):
        woeIv = self.get_woe_iv()
        for i,v in woeIv.items():
            maxWoe = v['woe'].max()
            minWoe = v['woe'].min()
            v['woe'] = (v['woe'] - minWoe) / (maxWoe - minWoe)
        return woeIv
    
    def woe_cal(self,data:pd.DataFrame,woeIv:dict):
        '''
        将data对应lable替换成对应的Woe
        '''
        lableUse = self.get_lable_use()
        missingMethod = self.missing
        
        for col in data.columns:
            
            if not lableUse[col]:
                continue
            ret,_ =self.__fix_missing_value_and_gen_data(data,col)
            data[col] = ret['X']
            if col in woeIv:
                bins = woeIv[col]['bin']
                woed = woeIv[col]['woe']
                intervals = [b for b in bins]
                for i, interval in enumerate(intervals):
                    if pd.api.types.is_interval(interval):
                        #区间的分箱
                        data.loc[(data[col] > interval.left) & (data[col] <= interval.right),col] = woed[i]
                    else:
                        #值的分箱
                        data.loc[data[col]==(interval),col]= woeIv[col]['woe'][i]
                       
                if missingMethod[col]['missingMethod']=='自成分箱': #空值处理 
                    #print(data[data[col].isnull()],col)
                    laDf:pd.DataFrame = woeIv[col]
                    res = laDf.loc[laDf['bin']=='Missing','woe']
                    if not res.empty:
                        res = res.item()
                        data[col] = data[col].fillna(res)
        return data
    
    def gen_linner_regression_model(self):
        useTable = self.gen_use_lable()
        woe = self.get_normalized_woe()
        data = self.woe_cal(self.traindata.copy(),woe)
        x_train = data.loc[:,useTable]
        y_train = data[self.ylable]
        lr = LogisticRegression()
        lr.fit(x_train,y_train)
        return lr
    
    def gen_use_lable(self ):
        return  [key for key, value in self.useLables.items() if value and key!=self.ylable]
        

        
    
        
    
if __name__ == "__main__":
    xsl = CardModelV1("output.xls")
    xsl.set_y_lable("是否返贫")
    xsl.set_missing_method('开支',"自成分箱","")
    xsl.get_lable_use()
    xsl.set_lable_use('Unnamed: 0',False)
    xsl.set_lable_use('姓名',False)
    data = pd.read_excel('test.xls')
    lrModel =  xsl.gen_linner_regression_model()
    data = xsl.woe_cal(data,xsl.get_normalized_woe())
    data.to_excel('woe.xls')
    px = data.loc[:,xsl.gen_use_lable()]
    py = data['是否返贫']
   
    pdy = lrModel.predict_proba(px)
    
    
   # print(pdy)

    