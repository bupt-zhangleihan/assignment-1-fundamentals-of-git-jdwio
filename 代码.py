# -*- coding: utf-8 -*-
"""

Get some chips

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt
import missingno
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rc('font', size=25)
# 数据预处理
filePath = r'insurance.csv'
dataBase = pd.read_csv(filePath)

# 查看缺失值
missingno.matrix(dataBase)

# bmi均值填充
dataBase['bmi'] = dataBase['bmi'].fillna(dataBase['bmi'].mean())
dataBase['children'] = dataBase['children'].fillna(dataBase['children'].mode()[0])
#%%
# 数据统计
for column in dataBase.columns:
    columnValue = dataBase[column]
    dtype = columnValue.dtype

    if dtype == 'float64' or dtype == 'int':
        plt.figure(figsize=(10, 8), dpi=100)
        sns.kdeplot(columnValue, fill=True, color='black')
        plt.title(column)
        plt.show()

    elif dtype == 'object':
        counter = Counter(columnValue)
        plt.figure(figsize=(10, 8), dpi=100)
        plt.pie(counter.values(), labels=counter.keys())
        plt.title(column)
        plt.show()
#%%
# 数据转换, 将文本变量转为类别数值
textFeatures = ['sex', 'smoker', 'region']
for feature in textFeatures:
    dataBase[feature] = dataBase[feature].astype('category').cat.codes
    
XNames = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

YName = 'charges'

# 数据加载
X = dataBase[XNames]
Y = dataBase[YName]

# 数据归一化
STD = StandardScaler()
X = STD.fit_transform(X)


# 按照8:2划分训练集和测试集
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=2023)

# 构建集成学习模型随机森林
RF = RandomForestRegressor(random_state=2023)

# 设置超参数
RFParams = {'n_estimators':[100, 150, 200, 300],
            'max_depth':[7, 8, 9, 10]}

# 网格搜索参数寻优
RFGrid = GridSearchCV(estimator=RF,
                      param_grid=RFParams,
                      scoring='r2',
                      cv=5,
                      verbose=50)
# 开始搜索
RFGrid.fit(xTrain, yTrain)
RFBestParams = RFGrid.best_params_
RF = RandomForestRegressor(n_estimators=RFBestParams['n_estimators'],
                           max_depth=RFBestParams['max_depth'])
RF.fit(xTrain, yTrain)

# 结果预测
RFPred = RF.predict(xTest)
# 计算指标
RFR2 = r2_score(yTest, RFPred)

#%%==========================================================================
DT = DecisionTreeRegressor(random_state=2023)
DTParams = {'max_depth':[7, 8, 9, 10]}
# 网格搜索参数寻优
DTGrid = GridSearchCV(estimator=DT,
                      param_grid=DTParams,
                      scoring='r2',
                      cv=5,
                      verbose=50)
# 开始搜索
DTGrid.fit(xTrain, yTrain)
DTBestParams = DTGrid.best_params_
DT = DecisionTreeRegressor(max_depth=DTBestParams['max_depth'])
DT.fit(xTrain, yTrain)

# 结果预测
DTPred = DT.predict(xTest)
# 计算指标
DTR2 = r2_score(yTest, DTPred)

#%%
Max = max(max(Y), max(RFPred), max(DTPred))
Min = min(min(Y), min(RFPred), min(DTPred))

Max = Max + abs(0.2 * Max)
Min = Min - abs(0.2 * Min)

plt.figure(figsize=(10, 8), dpi=100)
plt.scatter(yTest, RFPred, color='black', label='随机森林 $R^{2}$=' + '{:.4f}'.format(RFR2))
plt.scatter(yTest, DTPred, color='green', label='决策树 $R^{2}$=' + '{:.4f}'.format(DTR2))
plt.plot([Min,Max], [Min,Max], linestyle='dashed', color='royalblue')
plt.xlim(Min, Max)
plt.ylim(Min, Max)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.legend()
plt.show()


#%%
plt.figure(figsize=(10, 8),dpi=100)
nameScore = dict(zip(XNames, RF.feature_importances_ / sum(RF.feature_importances_)))
nameScore = dict(sorted(nameScore.items(),key = lambda x:x[1],reverse = False))
plt.barh(list(nameScore.keys()),list(nameScore.values()), color='black')
plt.xlabel('重要性分数')
plt.ylabel('特征')
plt.show()