import numpy as np
import pandas as pd

import math

from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

def cleanData(df):
    # Drop variables with little variance
    df = df.drop(['Id','Alley','Street', 'LotShape','Utilities', 'LandSlope','RoofMatl','Heating','Electrical','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','KitchenAbvGr', '3SsnPorch','ScreenPorch','PoolArea','PoolQC', 'MiscFeature', 'MiscVal'],axis=1)
    
    # Convert some integral classes to categorical
    df.MSSubClass = df.MSSubClass.astype('str')
    df.MoSold = df.MoSold.astype('str')
    df.YrSold = df.YrSold.astype('str')
    
    # Bin rare categories
    df.loc[df.MSSubClass.isin(['180','75','45','40','150']),'MSSubClass'] = 'Other'
    df.loc[df.MSZoning.isin(['RH', 'C (all)']),'MSZoning'] = 'Other'
    df.loc[df.Neighborhood.isin(['Blueste','NPkVill']),'Neighborhood'] = 'Other'
    df.loc[df.Condition1.isin(['RRAe','RRAn','RRNe','RRNn']),'Condition1'] = 'Near railroad'
    df.loc[df.Condition1.isin(['PosA','PosN']),'Condition1'] = 'Near positive feature'
    df.loc[df.Condition2.isin(['RRAe','RRAn','RRNe','RRNn']),'Condition2'] = 'Near railroad'
    df.loc[df.Condition2.isin(['PosA','PosN']),'Condition2'] = 'Near positive feature'
    df.loc[df.HouseStyle.isin(['2.5Unf','2.5Fin']),'HouseStyle'] = '2.5'
    
    def lengthMap(x):
        if x == 0 or math.isnan(x):
            area = 'None'
        else:
            area = str(x//50*50) + ' to ' + str((x//50+1)*50-1) + ' ft.' 
        return area
     
    df.LotFrontage = df.LotFrontage.apply(lambda x : lengthMap(x))
 
    def remodelAgeMap(x):
        if x == 1950: 
            era = 'No remodel'
        elif x > 1950 and x < 1960:
            era = '1950s'
        elif x >= 1960 and x < 1970:
            era = '1960s'
        elif x >= 1970 and x < 1980:
            era = '1970s'
        elif x >= 1980 and x < 1990:
            era = '1980s'
        elif x >= 1990 and x < 2000:
            era = '1990s'
        elif x >= 2000 and x < 2010:
            era = '2000s'
        else:
            era = '2010s'
        return era

    df.loc[:,'RemodelEra'] = df.YearRemodAdd.apply(lambda x : remodelAgeMap(x))
    df = df.drop('YearRemodAdd', axis = 1)
    
    df.loc[df.Exterior2nd.isin(['Wd Shng']),'Exterior2nd'] = 'WdShing'
    df.loc[df.Exterior2nd.isin(['CmentBd']),'Exterior2nd'] = 'CemntBd'
    df.loc[df.Exterior2nd.isin(['Brk Cmn']),'Exterior2nd'] = 'BrkComm'
    
    df.loc[df.RoofStyle.isin(['Flat','Gambrel','Mansard','Shed']),'RoofStyle'] = 'Other'
    df.loc[df.Exterior1st.isin(['AsphShn','ImStucc','CBlock','Stone','BrkComm']),'Exterior1st'] = 'Other'
    df.loc[df.Exterior2nd.isin(['AsphShn','ImStucc','CBlock','Stone','BrkComm']),'Exterior2nd'] = 'Other'
    
    def areaMap(x):
        if x == 0:
            area = 'None'
        else:
            area = str(x//50*50) + ' to ' + str((x//50+1)*50-1) + ' sq. ft.' 
        return area
    
    df.loc[:,'VeneerArea'] = df.MasVnrArea.apply(lambda x : areaMap(x))
    df = df.drop('MasVnrArea', axis = 1)
    
    df.loc[df.ExterCond.isin(['Po','Fa']),'ExterCond'] = 'Fa'
    df.loc[df.ExterCond.isin(['Gd','Ex']),'ExterCond'] = 'Gd'
    df.loc[df.Foundation.isin(['Wood','Stone','Slab']),'Foundation'] = 'Other'
    df.loc[df.BsmtCond.isin(['Po','Fa']),'BsmtCond'] = 'Fa'
    
    df.loc[:,'BasementUnfinishedSF'] = df.BsmtUnfSF.apply(lambda x : areaMap(x))
    df = df.drop('BsmtUnfSF',axis = 1)
    
    df.loc[:,'TotalBasementSF'] = df.TotalBsmtSF.apply(lambda x : areaMap(x))
    df = df.drop('TotalBsmtSF', axis = 1)
    
    df.loc[df.HeatingQC.isin(['Po','Fa']),'HeatingQC'] = 'Fa'
    
    df.loc[:,'TotalIndoorSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df = df.drop(['1stFlrSF','2ndFlrSF'],axis=1)
    
    df.loc[:,'TwoBasementFullBath'] = df.BsmtFullBath.apply(lambda x : 'Yes' if x == 2 else 'No')
    df = df.drop('BsmtFullBath',axis=1)
    
    df.loc[:,'TwoHalfBath'] = df.HalfBath.apply(lambda x : 'Yes' if x == 2 else 'No')
    df = df.drop('HalfBath',axis=1)
    
    df.loc[df.Functional.isin(['Maj1','Maj2','Sev']),'Functional'] = 'Other'
    df.loc[df.Functional.isin(['Min1','Min2']),'Functional'] = 'Minimial'
    
    df.loc[df.GarageType.isin(['CarPort','2Types']),'GarageType'] = 'Other'

    df.GarageArea = df.GarageArea.apply(lambda x : areaMap(x))
    df.loc[df.GarageQual.isin(['Ex','Gd']),'GarageQual'] = 'Gd'
    df.loc[df.GarageQual.isin(['Po','Fa']),'GarageQual'] = 'Fa'
    
    df.loc[df.GarageCond.isin(['Ex','Gd']),'GarageCond'] = 'Gd'
    df.loc[df.GarageCond.isin(['Po','Fa']),'GarageCond'] = 'Fa'
    
    df.WoodDeckSF = df.WoodDeckSF.apply(lambda x : areaMap(x))
    df.OpenPorchSF = df.OpenPorchSF.apply(lambda x : areaMap(x))
    df.EnclosedPorch = df.EnclosedPorch.apply(lambda x : areaMap(x))
    
    df.loc[df.Fence.isin(['MnWw']),'Fence'] = 'MnPrv'
    
    df.loc[df.SaleType.isin(['Con','Oth','CWD','ConLI','ConLw','ConLD']),'SaleType'] = 'Other'
    df.loc[df.SaleCondition.isin(['AdjLand','Alloca']),'SaleCondition'] = 'Other'
    
    # Impute missing values with a "None" feature or a computed feature
    df.loc[df.Fence.isna(),'Fence'] = 'None'
    df.loc[df.FireplaceQu.isna(), 'FireplaceQu'] = 'None'
    df.loc[df.GarageCond.isna(),'GarageCond'] = 'None'
    df.loc[df.GarageYrBlt.isna(),'GarageYrBlt'] = 'None'
    df.loc[df.GarageFinish.isna(),'GarageFinish'] = 'None'
    df.loc[df.GarageQual.isna(),'GarageQual'] = 'None'
    df.loc[df.GarageType.isna(),'GarageType'] = 'None'
    df.loc[df.BsmtCond.isna(),'BsmtCond'] = 'None'
    df.loc[df.BsmtExposure.isna(),'BsmtExposure'] = 'None'
    df.loc[df.BsmtQual.isna(),'BsmtQual'] = 'None'
    df.loc[df.BsmtFinType1.isna(),'BsmtFinType1'] = 'None'
    df.loc[df.MSZoning.isna(),'MSZoning'] = 'Other'
    df.loc[df.Functional.isna(),'Functional'] = 'Other'
    df.loc[df.SaleType.isna(),'SaleType'] = 'Other'
    df.loc[df.KitchenQual.isna(),'KitchenQual'] = df.groupby('KitchenQual').KitchenQual.count().sort_values(ascending = False).index[1]
    df.loc[df.GarageCars.isna(),'GarageCars'] = 0
    df.loc[df.Exterior1st.isna(),'Exterior1st'] = 'Other'
    df.loc[df.Exterior2nd.isna(),'Exterior2nd'] = 'Other'
    df.loc[df.MasVnrType.isna(),'MasVnrType'] = 'None'
    
    # Standardize all numeric variables
    df = df.apply(lambda x : scale(x) if x.dtype.name != 'object' and x.name != 'SalePrice' else x)
    
    # One hot encode categoricals
    df = pd.get_dummies(df)
    
    table = {'Condition' : ['Norm', 'Feedr', 'Near positive feature', 'Artery','Near railroad'],
                'Exterior' : ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing','CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'Other']}

    # Combine Exterior1st and Exterior2nd features
    # Combine Condition1 and Condition2 features
    def transformCols(row):        
        for name in table['Condition']:
            row['Condition' + '_' + name] = max(row['Condition1_' + name],row['Condition2_' + name])
        
        for name in table['Exterior']:
            row['Exterior' + '_' + name] = max(row['Exterior1st_' + name],row['Exterior2nd_' + name])
        
        return row
    
    df = df.transform(transformCols,axis=1) 
    
    for name in table['Condition']:
        df.drop(['Condition1_' + name,'Condition2_' + name], axis=1, inplace = True)
        
    for name in table['Exterior']:
        df.drop(['Exterior1st_' + name,'Exterior2nd_' + name], axis = 1, inplace = True)    

    df.SalePrice = np.log(df.SalePrice)
    
    return [df[df.Type_train == 1], df[df.Type_test == 1]]

# Dump train outliers
train = train.loc[train['LotArea'] < 100000,:]

train.loc[:,'Type'] = 'train'
test.loc[:,'Type'] = 'test'

combined = pd.concat([train, test], axis = 0,  sort = False)

train_clean, test_clean = cleanData(combined)

train_clean.drop(['Type_train', 'Type_test'], axis=1, inplace=True)
test_clean.drop(['Type_train', 'Type_test'], axis=1, inplace=True)

X = train_clean.drop('SalePrice', axis=1)
y = train_clean.SalePrice

# Models - Lasso
lasso = Lasso()
param_grid = { 'alpha': np.linspace(0.0001,0.1, 300), 'max_iter' : [10000]}
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

print(grid_search.best_params_, math.sqrt(math.fabs(grid_search.best_score_)))

