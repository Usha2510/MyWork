#!/usr/bin/env python
# coding: utf-8

# # Pump it Up - DataDriven Challenge
Index
1. Loading data

2. Exploratory Data Analysis and Understanding
    2.1 Data Distribution and Statistics
    2.2 Data Visualizations on Category Features
    2.3 Advanced Data Exploration
    2.4 Unecessary Features
    2.5 Data Correctness
    2.6 Missing values

3. Feature Engineering

4. Feature Selection
    4.1 Information gain
    4.2 LogisticRegression L1 selection

5. Modeling and Evaluation
    5.1 Random Forest
        5.1.1 Random Forest without PCA    
        5.1.2 Random Forest with PCA  
    5.2 XGBoost
    5.3 LDA
    5.4 LightGBM
    5.5 Voting Classsifier

# In[1]:


#Basics
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean, std

from geopy  import distance
from datetime import datetime
from collections import Counter
#Train Test Split
from sklearn.model_selection import train_test_split

# Imputer
from sklearn.impute import SimpleImputer

# Preprocessing
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing, metrics 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler, 
    FunctionTransformer,MinMaxScaler)
from category_encoders import TargetEncoder, CountEncoder

from category_encoders.wrapper import PolynomialWrapper
#Stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
#Grid Search
from sklearn.model_selection import GridSearchCV
# Classifiers
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
#from catboost import CatBoostClassifier
import lightgbm as lgbm  
import xgboost as xgb
from sklearn.metrics import log_loss
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier

#Clusters
from sklearn.cluster import KMeans

# Model evaluation
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
# requires separate installation

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')
sns.set(font_scale=2)
SEED = 42


# In[2]:


#Converting Catgeorical Variables to strings
def convert_categorical_to_string(df):
    return df.astype(str)



##Classifying Variables
def classify_columns(df):
    """Takes a dataframe and a list of columns to 
    drop and returns:
        - cat_cols: A list of categorical columns.
        - num_cols: A list of numerical columns.
    """
    cols = df.columns
    cat_cols = []
    num_cols = []
    for col in cols:
        if col != "status_group":
            if df[col].dtype == object :
                cat_cols.append(col)
            else:
                num_cols.append(col)
    return cat_cols, num_cols


#Dropping features
def drop_cols(df, features):
    for feature in features:
        df.drop(feature, axis=1, inplace=True)
        
        
#Numerical Imputation
def num_imp(df,num_missingfeatures_list,measure):

    for feature in num_missingfeatures_list:
        df[feature].fillna(df.groupby(['region', 'district_code'])[feature].transform(measure), inplace=True)
        df[feature].fillna(df.groupby(['region'])[feature].transform(measure), inplace=True)
        df[feature].fillna(df[feature].median(), inplace=True)
        
        
#Catgeorical Imputation
def cat_imp(df,cat_missingfeatures_list,value): 
    for feature in cat_missingfeatures_list:
        df[feature].fillna(value, inplace=True)

def top_unique(df, features):
    col_unique_list = []
    column_list = []
    for col in features:
        column_list.append(col) 
        col_unique_list.append(df[col].sort_values(ascending=False).unique().tolist()[1:50])        
    data_tuples = list(zip(column_list, col_unique_list))
    keywords = pd.DataFrame(data_tuples, columns = ['column_name', 'column_uniques'])
    return keywords


def most_frequent(df,features):
    col_unique_list = []
    column_list = []
    for col in features:
        c = Counter(df[col])
        common_list = c.most_common(50)
        column_unique_values = [r[0] for r in common_list]
        column_list.append(col)
        col_unique_list.append(column_unique_values)
    data_tuples = list(zip(column_list, col_unique_list))
    keywords = pd.DataFrame(data_tuples, columns = ['column_name', 'column_uniques'])    
    return keywords    
        
        
#Target Encoding
def target_enc(train_df, test_df, features):
    targ_enc = TargetEncoder(cols=features)
    targ_enc.fit(train_df[features], train_df['status_group'])
    
    train_df = train_df.join(targ_enc.transform(train_df[features]).add_suffix('_targ'))
    test_df = test_df.join(targ_enc.transform(test_df[features]).add_suffix('_targ'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df

    
#Target Encoding
def target_enc_train(train_df, features):
    targ_enc = TargetEncoder(cols=features)
    targ_enc.fit(train_df[features], train_df['status_group'])
    
    train_df = train_df.join(targ_enc.transform(train_df[features]).add_suffix('_targ'))
   
    train_df = train_df.drop(features, axis=1)
   
    
    return train_df
##One Hot Encoding
#One hot Encoding
def onehot_enc(train_df, features):
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(train_df[features])
    
    _ohe_array = enc.transform(train_df[features])
    _ohe_names = enc.get_feature_names()
    for i in range(_ohe_array.shape[1]):
      train_df[_ohe_names[i]] = _ohe_array[:,i]
    
    
    return train_df


#Scaling
def standardization(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    
#data_clean = data.copy()
#for col in topNlist:
   # values = keywords.loc[keywords['column_name'] == col, 'column_uniques']
    
    #for i, row in data_clean.iterrows():
        
      #  if ifor_val not in values:
          #  ifor_val = "Others"
       # data_clean.at[i,col] = ifor_val
                


# # Modules

# In[3]:


class Govtcorrection(BaseEstimator, TransformerMixin):
    """ Text
        
    Args: 
        name (type): description
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame with new features
    """
    
    def __init__(self, installer=False, funder=False):
        self.installer = installer
        self.funder = funder
    
    def fit(self, X, y=None):    
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try:
        
            if self.installer:
                gov_installer = X.installer.str.lower().str.slice(stop=5).isin(['gover', 'centr','tanza', 'cetra'])
                X.loc[gov_installer, 'installer'] = 'government'
                X.loc[:, 'installer'] = X.apply(lambda row: row.installer.lower()[:4] if pd.notnull(row.installer) else row, axis=1) 
            
            if self.funder:
                gov_funder = X.funder.str.lower().str.slice(stop=5).isin(['gover', 'centr','tanza', 'cetra'])
                X.loc[gov_funder, 'funder'] = 'government'
                X.loc[:, 'funder'] = X.apply(lambda row: row.funder.lower()[:4] if pd.notnull(row.funder) else row, axis=1) 
                
            return X
            
        except KeyError:
            cols_related = ['installer', 'funder']
            
            cols_error = list(set(cols_related) - set(X.columns))
            raise KeyError('[Govtcorrection] DataFrame does not include the columns:', cols_error)


# In[4]:


class Distance_from_capital(BaseEstimator, TransformerMixin):
    """Feature creation based on existing geographic variables
    
    Args:
        distance_to_Dodoma (bool): if True creates manhattan distance from the waterpoint to Dodoma, default True
        distance_to_Salaam (bool): if True creates manhattan distance from the waterpoint to Salaam, default True
        strategy (str): 'manhattan' or 'eucledian' distance
    
    Returns: 
        pd.DataFrame: transformed pandas DataFrame.
    """
    
    def __init__(self, distance_to_Dodoma=True, distance_to_Salaam=True, strategy='manhattan'):
        self.distance_to_Dodoma = distance_to_Dodoma
        self.distance_to_Salaam = distance_to_Salaam
        self.strategy = strategy
        
    def fit(self,X,y=None):    
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try: 
            dodoma = (-6.1630, 35.7516)
            salaam = (-6.7924, 39.2083)

            if self.strategy == 'manhattan':
                if self.distance_to_Dodoma:
                    X.loc[:, 'distance_to_Dodoma'] = np.abs(X.loc[:,'longitude'] - dodoma[1]) + np.abs(X.loc[:, 'latitude'] - dodoma[0])

                if self.distance_to_Salaam:
                    X.loc[:, 'distance_to_Salaam'] = np.abs(X.loc[:,'longitude'] - salaam[1]) + np.abs(X.loc[:, 'latitude'] - salaam[0])
            
            elif self.strategy == 'eucledian':
                if self.distance_to_Dodoma:
                    X.loc[:, 'distance_to_Dodoma'] = np.sqrt((X.loc[:,'longitude'] - dodoma[1])**2 + (X.loc[:, 'latitude'] - dodoma[0])**2)
                
                if self.distance_to_Salaam:
                    X.loc[:, 'distance_to_Salaam'] = np.sqrt((X.loc[:,'longitude'] - salaam[1])**2 + (X.loc[:, 'latitude'] - salaam[0])**2)
                    
            else:
                raise KeyError('Strategy is wrong. Should be either "manhattan" or "eucledian"')
            
            return X

        except KeyError:
            cols_error = list(set(['longtitude', 'latitude']) - set(X.columns))
            raise KeyError('[Distance_from_capital] DataFrame does not include the columns:', cols_error)
            


# In[5]:


class DropColumns(BaseEstimator, TransformerMixin):
    """ Drops columns specified and returns transformed DataFrame
    
    Args:
        columns (List): list of column names to drop
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame.
        
    """ 
    
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X.drop(columns=self.columns)
        
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("[DropCol] DataFrame does not include the columns: %s", cols_error)
            


# In[6]:


class GeoClustering(BaseEstimator, TransformerMixin):
    """Text
        
    Args: 
        name (type): description
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame with new features
    """
    
    def __init__(self, n_clusters=50):
        self.n_clusters = n_clusters
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state = 289))
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X[['longitude','latitude']])             
            
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try:
            labels = self.pipeline.predict(X[['longitude','latitude']]).astype('str')
            X = X.assign(cluster = labels)
            
            return X
            
        except KeyError:
            cols_related = ['longitude','latitude']
            
            cols_error = list(set(cols_related) - set(X.columns))
            raise KeyError('[Clustering] DataFrame does not include the columns:', cols_error)
        


# In[7]:


class Interactions(BaseEstimator, TransformerMixin):
    """Feature creation based on existing categorical variables, interactions between them
    
    Args:
        scheme_management_payment (bool): if True creates feature interaction of scheme_management and payment (scheme_management + payment), default True
        basin_source (bool): if True creates feature interaction of basin and source (basin + source), default True
        source_waterpoint_type (bool): if True creates feature interaction of source and waterpoint_type (source + waterpoint_type), default True
        extraction_waterpoint_type (bool): if True creates feature interaction of extraction and waterpoint_type (extraction + waterpoint_type), default 
        True
        water_quality_quantity (bool): if True creates feature interaction of water_quality and quantity (water_quality + quantity), default True
        source_extraction_type (bool): if True creates feature interaction of source and extraction_type (source + extraction_type), default True
        extraction_type_payment (bool): if True creates feature interaction of payment and extraction_type (payment + extraction_type), default True
           
    Returns: 
        pd.DataFrame: transformed pandas DataFrame.
    """
    
    def __init__(self, scheme_management_payment=True, basin_source=True, 
                 source_waterpoint_type=True, extraction_waterpoint_type=True, 
                 water_quality_quantity=True, source_extraction_type=True, extraction_type_payment=True):
        self.scheme_management_payment = scheme_management_payment
        self.basin_source = basin_source
        self.source_waterpoint_type = source_waterpoint_type
        self.extraction_waterpoint_type = extraction_waterpoint_type
        self.water_quality_quantity = water_quality_quantity
        self.source_extraction_type = source_extraction_type
        self.extraction_type_payment = extraction_type_payment
 
    
    def fit(self,X,y=None):    
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try: 

            if self.scheme_management_payment:
                X.loc[:, 'scheme_management_payment'] = X.loc[:,'scheme_management'] + '_' + X.loc[:,'payment']
                                                           
            if self.basin_source:
                X.loc[:, 'basin_source'] = X.loc[:,'basin'] + '_' + X.loc[:,'source']
            
            if self.source_waterpoint_type:
                X.loc[:, 'source_waterpoint_type'] = X.loc[:,'source'] + '_' + X.loc[:,'waterpoint_type']
            
            if self.extraction_waterpoint_type:
                X.loc[:, 'extraction_waterpoint_type'] = X.loc[:,'extraction_type'] + '_' + X.loc[:,'waterpoint_type']    
            
            if self.source_extraction_type:
                X.loc[:, 'source_waterpoint_type'] = X.loc[:,'source'] + '_' + X.loc[:,'extraction_type']
            
            if self.water_quality_quantity:
                X.loc[:, 'water_quality_quantity'] = X.loc[:,'water_quality'] + '_' + X.loc[:,'quantity']
            
            if self.extraction_type_payment:
                X.loc[:, 'extraction_type_payment'] = X.loc[:,'extraction_type'] + '_' + X.loc[:,'payment']
                
            return X                                                                                                                     
                                                                      
        except KeyError:
            cols_error = list(set(['scheme_management', 'basin', 'source', 'population', 'payment', 'waterpoint_type', 'extraction_type', 'water_quality', 'quantity' ]) - set(X.columns))
            raise KeyError('[Interactions] DataFrame does not include the columns:', cols_error)
            
    


# In[8]:


class NewFeatures(BaseEstimator, TransformerMixin):
    """Feature creation based on existing variables
    
    Feature "dry_season" was implemented based on  the following research: 
    https://lib.ugent.be/fulltxt/RUG01/002/350/680/RUG01-002350680_2017_0001_AC.pdf
    Dry season covers the following months: January(1), February(2), June(6), July(7), August(8), September(9), October(10)
    Wet season covers the following months: March(3), April(4), May(5), November(11), December(12)
    
    Args:
        type_wpt_name (bool): if True creates feature representing type of the waterpoint, default True
        water_per_capita (bool): if True creates feature ratio of amount_tsh to population (amount_tsh / population), default True
        dry_season (bool): if True creates the feature representing if the season is dry (1), 0 otherwise (if it's wet season)
        num_private (bool): if True transforms num_private into True/False feature
        age (bool): if True creates new feature representing the difference between date recorded and construction year
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame.
    """
    
    def __init__(self, type_wpt_name=True, water_per_capita=True, dry_season=True, num_private=True, age=True):
        self.type_wpt_name = type_wpt_name
        self.water_per_capita = water_per_capita
        self.dry_season = dry_season
        self.num_private = num_private
        self.age = age
        
    def fit(self,X,y=None):    
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try: 

            if self.age:
                X.loc[:,'date_recorded'] = pd.to_datetime(X.date_recorded)
                X.loc[:, 'age'] = X.loc[:, 'date_recorded'].dt.year - X.loc[:, 'construction_year'] 
            if self.type_wpt_name:
                X.loc[:, 'type_wpt_name'] = X.loc[:,'wpt_name'].apply(lambda x: x.split(' ')[0].strip()).replace('Zahanati-Misssion', 'Zahanati')
    
            if self.water_per_capita:
                X.loc[:, 'water_per_capita'] = X.loc[:,'amount_tsh'] / (X.loc[:, 'population'] + 1)
            
            if self.dry_season:
                
                X.loc[:,'date_recorded'] = pd.to_datetime(X.date_recorded)
                X = X.assign(dry_season = X.date_recorded.dt.month)
                X.dry_season = X.dry_season.replace([1,2,3,4,5,6,7,8,9,10,11,12],[1,1,0,0,0,1,1,1,1,1,0,0])
            
            if self.num_private:
                X.loc[:, "num_private"] = X.loc[:, 'num_private'].ne(0).astype(int)
                
            
                
            return X


        except KeyError:
            cols_error = list(set(['wpt_name', 'amount_tsh', 'population', 'num_private', 'date_recorded']) - set(X.columns))
            raise KeyError('[NewFeatures] DataFrame does not include the columns:', cols_error)


# In[9]:


class OurAdvancedImputer(BaseEstimator, TransformerMixin):
    """Custom advanced imputation of missing values
        
    Args: 
        name (type): description
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame with new features
    """
    
    def __init__(self, population_bucket=True):
        self.population_bucket = population_bucket
        self.cluster_population = {}
    
    def fit(self, X, y=None):
        X.loc[X.population.isin([0,1]), 'population'] = np.nan
        for cluster in X.cluster.unique():
            population_mean = X.loc[X.cluster==cluster, 'population'].mean()
            if np.isnan(population_mean):
                self.cluster_population[cluster] = 0
            else:
                self.cluster_population[cluster] = population_mean 
        
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try:
            X.loc[X.population.isin([0,1]), 'population'] = np.nan
            X.loc[X.population.isnull(),'population'] = X.apply(lambda row: self.cluster_population.get(row.cluster), axis=1)
            
            if self.population_bucket:
                population_log = np.log1p(X.population)
                population_log_binned = pd.cut(population_log, bins=[0,2,6,np.inf], include_lowest=True, labels=[1,2,3])
                X = X.assign(population_binned = population_log_binned)
                                
            return X
            
        except KeyError:
            cols_related = ['population']
            
            cols_error = list(set(cols_related) - set(X.columns))
            raise KeyError('[OurAdvancedImputer] DataFrame does not include the columns:', cols_error)


# In[10]:


class OurSimpleImputer(BaseEstimator, TransformerMixin):
    """Custom imputation of missing values
        
    Args: 
        name (type): description
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame with new features
    """
    
    def __init__(self, categorical=True,coords=True,permit=True, construction_year=True):
        self.permit = permit
        self.categorical = categorical
        self.coords = coords
        self.lga_coords = {}
        self.construction_year = construction_year
        self.extraction_dict = {}
    
    def fit(self, X, y=None):
        # saving center coordinates of each LGA
        if self.coords:
            X.loc[X.longitude == 0, ['longitude','latitude']] = np.nan
            for lga in X.lga.unique():
                if lga == 'Geita':                    
                    lat = -2.869440
                    lon = 32.234906
                else:
                    lat = X.loc[X.lga == lga, 'latitude'].mean()
                    lon = X.loc[X.lga == lga, 'longitude'].mean()
                self.lga_coords[lga] = (lat, lon)
        
        if self.construction_year:
            X.loc[X.construction_year.isin([0]), 'construction_year'] = np.nan
            for cluster in X.extraction_type.unique():
                year_mean = X.loc[X.extraction_type==cluster, 'construction_year'].median()
                if np.isnan(year_mean):
                    self.extraction_dict[cluster] = 0
                else:
                    self.extraction_dict[cluster] = year_mean
        
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try:
            
            # categorical features
            if self.categorical:
                na_names = ['Not known', 'not known', '-', 'No', 'no', 'Unknown', '0', 'none']
                X = X.replace(na_names, np.nan)
                X = X.fillna({
                    'funder': 'unknown', 
                    'installer': 'unknown', 
                    'management': 'unknown', 
                    'payment': 'unknown', 
                    'water_quality': 'unknown', 
                    'quantity': 'unknown', 
                    'source': 'unknown',
                    'wpt_name': 'unknown',
                    'scheme_management': 'unknown',
                    'permit': self.permit
                })
                X.loc[:, 'permit'] = X.loc[:, 'permit'].astype(int)
            
            # coordinates
            if self.coords:
                X.loc[X.longitude==0, ['longitude', 'latitude']] = np.nan
                X.loc[X.latitude.isnull(),'latitude'] = X.apply(lambda row: self.lga_coords.get(row.lga)[0], axis=1)
                X.loc[X.longitude.isnull(),'longitude'] = X.apply(lambda row: self.lga_coords.get(row.lga)[1], axis=1)
                
            # construction year
            if self.construction_year:
                X.loc[X.construction_year==0, 'construction_year'] = np.nan
                X.loc[X.construction_year.isnull(),'construction_year'] = X.apply(lambda row: self.extraction_dict.get(row.extraction_type), axis=1)
                    
            return X
            
        except KeyError:
            cols_related = ['funder', 'installer', 'management', 'payment',
                            'water_quality', 'quantity', 'source', 'scheme_management',
                           'latitude','longitude','lga', 'construction_year']
            
            cols_error = list(set(cols_related) - set(X.columns))
            raise KeyError('[OurImputer] DataFrame does not include the columns:', cols_error)
        
        


# In[11]:


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.

    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target 
            encode all categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X 
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)


# # Loading Data

# In[12]:




path = "D:/MMA - Smith School of Business/8 Machine Learning & AI - MMA 869/Team Project/raw/"
output_path = "D:/MMA - Smith School of Business/8 Machine Learning & AI - MMA 869/Team Project/Output/"

original_train_labels  = pd.read_csv(path + "Training_set_Y.csv")

original_train_values  = pd.read_csv(path + 'Training_set_X.csv')
test  = pd.read_csv(path + 'Test_set.csv')
original_train = pd.merge(original_train_values, original_train_labels, on='id')
original_train.date_recorded = pd.to_datetime(original_train.date_recorded)


# In[13]:


train = original_train.copy()


# In[14]:


cat_cols , num_cols = classify_columns(train)


# In[33]:


len(cat_cols)


# In[73]:


train["population"].ihst


# # 1. Exploratory Data Analysis (EDA)

# # 1.1 Data Distribution and Statistics

# Let us look at the distribution of target variable

# In[14]:


#plot the target variable to view
plt.subplots(figsize=(10,6))
ax = sns.countplot(x=train['status_group'])
for p in ax.patches:
        ax.annotate('{:.2f}%'.format(p.get_height()*100/len(train)), (p.get_x()+0.3, p.get_height()*0.5))


# There are much more pumps that are functional and non-functional in our dataset and only a mere 7.27% of our dataset contains pumps that are functional but need repair. There is a class imbalance in the dataset

# # Correlation Matrix

# In[15]:


plt.figure(figsize=(18,18))
corr_df = train.corr()
numerical_features = ['id','num_private','region_code','district_code','construction_year']
heatmap_df=corr_df.drop(numerical_features).drop(numerical_features,axis=1)

sns.heatmap(heatmap_df,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 250, s=80, l=55, n=9)
)


# In[59]:


# Correlation Matrix Heatmap Visualization (should run this code again after removing outliers/zero values)
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure to control size of heatmap
fig, ax = plt.subplots(figsize=(12,8))
# Create a custom color palette
cmap = sns.diverging_palette(285, 10, as_cmap=True)  # as_cmap returns a matplotlib colormap object rather than a list of colors
# Red=10, Green=128, Blue=255
# Plot the heatmap
sns.heatmap(train.corr(), mask=mask, annot=True, square=True, cmap=cmap , vmin=-1, vmax=1, ax=ax)  # annot display corr label
# Prevent Heatmap Cut-Off Issue
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)


# Population and gps_height are also slightly correlated.

# # 1.2 Data Visualizations on Category Features

# Water_quality and quality_group 

# In[35]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='population')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.population.value_counts()


# In[101]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='quality_group')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.quality_group.value_counts()


# quantity and quantity_group

# In[21]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='quantity')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.quantity.value_counts()


# In[22]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='quantity_group')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.quantity_group.value_counts()


# waterpoint_type and waterpoint_type_group

# In[23]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='waterpoint_type')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.waterpoint_type.value_counts()


# In[24]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='waterpoint_type_group')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.waterpoint_type_group.value_counts()


# payment and payment_type

# In[25]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='payment')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.payment.value_counts()


# In[26]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='payment_type')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.payment_type.value_counts()


# management and management_group

# In[27]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='management')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.management.value_counts()


# In[28]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='management_group')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.management_group.value_counts()


# scheme_management and scheme_name

# In[29]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='scheme_management')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.scheme_management.value_counts()


# In[30]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='scheme_name')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.scheme_name.value_counts()


# source, source_type and source_class

# In[31]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='source')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.source.value_counts()


# In[32]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='source_type')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.source_type.value_counts()


# In[33]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='source_class')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.source_class.value_counts()


# extraction_type and extraction_type_group and extraction_type_class

# In[34]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='extraction_type')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.extraction_type.value_counts()


# In[35]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='extraction_type_group')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.extraction_type_group.value_counts()


# In[36]:


plt.figure(figsize=(14,6))
sns.countplot(data=train,x='extraction_type_class')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
train.extraction_type_class.value_counts()


# In[107]:


train.groupby(['source_class', 'source_type', 'source']).size()


# In[38]:


train.groupby(['payment', 'payment_type']).size()


# In[39]:


train.groupby(['water_quality', 'quality_group']).size()


# In[40]:


train.groupby(['quantity', 'quantity_group']).size()


# In[41]:


train.groupby(['waterpoint_type', 'waterpoint_type_group']).size()


# In[42]:


train.groupby(['scheme_management', 'scheme_name']).size()


# In[43]:


train.groupby(['extraction_type', 'extraction_type_group', 'extraction_type_class']).size()


# In[44]:


train.groupby(['management' , 'management_group']).size()


# In[45]:


train.nunique().sort_values()


# # Unnecessary Features

# There are few hierarchial columns that contain the same information but on different granularity level. 
# We decided to keep the columns with the lowest granular level:
# 1. waterpoint_type / waterpoint_type_group
# 2. source / source_type / source_class
# 3. quantity / quantity_group
# 4. water_quality / quality_group
# 5. payment / payment_type
# 6. management / management_group
# 7. extraction_type / extraction_type_group / extraction_type_class
# 8. scheme_management / scheme_name

# We have also decided to drop the below 7 features due to the following reasons:
#     1. id: surrogate key, does not contain any information
#     2. recorded_by : contains information about organization that recorded the data and only has one value
#     3. district code, region_code: Same as long and lat
#     4. region: lga has more granular level
#     5. subvillage:Lat long can be used instead
#     6. Public_Meeting: It has only two unique values, According to research on water issue in Tanzania, public meeting at installation is not good predictor

# # Amount_tsh

# In[108]:


plt.figure(figsize = (10,10))
ax = sns.boxplot(x="status_group", y="amount_tsh", data=train.loc[(train.amount_tsh != 0)])


# Plotting the data as is, gives us an idea that the distribution of the data spread across functionality is heavily skewed and uneven due to the large number of outliers.

# # Geographic Location wise Analysis

# In[18]:


# A bar chart showing the population of each region
df_pop = train.dropna(subset=['region'])
df_pop = train.loc[:, ['region', 'population']]

fig = plt.gcf()
fig.set_size_inches(12, 12)

ax = sns.countplot(data=df_pop, x='region')

ax.set(xlabel='Region Name', ylabel='Population', title='Population split by Region')

sns.set(font_scale=1.7)
plt.xticks(rotation=45);


# # Water Pumps Status based on Region

# In[112]:


# A bar chart showing the number of each status group by each region
df_region = train.dropna(subset=['region'])
df_region = train.loc[:, ['region', 'status_group']]

fig = plt.gcf()
fig.set_size_inches(20, 10)

ax = sns.countplot(data=df_region, x='region', hue='status_group')

ax.set(xlabel='Region Name', ylabel='Number of Water Pumps', title='Pump Status and Number of Water Pumps per Region')


plt.xticks(rotation=45);


# From the chart above we can see that Dodoma - the official capital and Dar es Salaam - the commercial capital of Tanzania, have comparatively lower number of water pumps for each category of functionality compared to other regions. This could imply that these more developed regions rely on piping and have a generally better water infrastructure and therefore don't require as many water pumps.

# # Drop columnns

# In[49]:


drop_features = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

train = train.drop(drop_features, axis=1)
test_orig = test.drop(drop_features, axis=1)


# In[50]:


train.info()


# In[36]:


cat_cols, num_cols = classify_columns(train)
cat_cols


# # Correcting Installer and Funder

# In[ ]:





# In[52]:


train = Govtcorrection(installer=True, funder=True).fit_transform(train)


# # Correcting construction year where it is greater than date recorded

# In[53]:


train['date_recorded'] = pd.to_datetime(train['date_recorded'], format='%Y-%m-%d')


# In[54]:


train[train.date_recorded.dt.year < train.construction_year][['construction_year', 'date_recorded']]


# Impute the construction year with 0 where date recorded is less than construction year

# In[55]:


train.loc[train.date_recorded.dt.year < train.construction_year,'construction_year'] = 0


# In[56]:


train[train.date_recorded.dt.year < train.construction_year][['construction_year', 'date_recorded']]


# # Missing Values

# In[57]:


data= train.copy()
data['amount_tsh'].replace(0.0, np.nan, inplace=True)
data['gps_height'].replace(0, np.nan, inplace=True)
data['num_private'].replace(0, np.nan, inplace=True)
data['population'].replace(0, np.nan, inplace=True)
data['construction_year'].replace(0, np.nan, inplace=True)
data.isnull().sum()


# Replacing Missing values with unknown:
# - Values can be [nan, 'Not known', 'not known', '-', 'No', 'no', 'Unknown', '0', 'none']
# This is handled by OurSimpleImputer

# In[58]:


train = OurSimpleImputer(coords=False).fit_transform(train)


# # Construction Year

# This feature contains value of 0, which indicates missing value, this has been dealt with by imputing the median value for that particular extraction_type. This process is taken care of with the use of the OurSimpleImputer transformation.

# # Latitude and Longitude

# After checking longitude and latitude, we found that we have some of the values to be 0. Given that this is Tanzania, latitude and longitude of 0 or close to 0 are simply impossible. We decided to replace those values with mean for each LGA, which has quite detailed granularity (125 LGAs).

# In[59]:


train = OurSimpleImputer(categorical=False).fit_transform(train)


# # Population

# Almost 48% of observations have value "population" equal to 0 or 1. According to the description, this feature indicates the population around the particular waterpoint. It is hard to believe that the waterpoint was built for 0 or 1 person. Hence, we believe these two values indicate missing values or incorrect recording.  Given that this is a population around a particular waterpoint, we decided to use population average of each coordinates-based cluster to impute missing values.

# (Seems like 50 is good number of K-Means clusters)

# In[60]:


locations = train[['latitude','longitude']]
locations_transformed = StandardScaler().fit_transform(locations)

n_clusters = [10, 20, 30, 40, 60, 80, 100, 150]
inertia = []

for num in n_clusters:
    kmeans_model = KMeans(n_clusters=num, n_jobs=-1).fit(locations_transformed)
    wss = kmeans_model.inertia_
    inertia.append(wss)
    
pd.DataFrame({'n_clusters':n_clusters, 'inertia': inertia}).plot(x='n_clusters', y='inertia');


# In[61]:


#train = GeoClustering().fit_transform(X)
#train = OurAdvancedImputer().fit_transform()


# # 2 Feature Engineering
1. Distance from capital for each lat , long (manhattan or euclidean)
2. Clustering based on the geo location
3. Replace num_private to True/False
4. Tagging Dry/wet season (based on months) from date_recorded (True/False)
5. Interactions:
    a. scheme_management + payment
    b. basin + source
    c. source + waterpoint_type
    d. extraction_type + waterpoint_type
    e. source + extraction_type
    f. water_quality + quantity
    g. extaction_type + payment
6. CountEncoder only: funder, installer, wpt_name, lga, ward
7. 1Hot / CountEncoder: 
   basin(9), scheme_management(13), management(12), source(10), payment(7), extraction_type(18), waterpoint_type(7),
   water_quality(8), quantity(5)
8. Age feature: year the data was recorded - construction year
9. Bucketization for population
# In[62]:


irrelevant_features = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

drop_features = ['latitude', 'longitude', 'date_recorded', 'num_private']

transformation_pipeline = Pipeline([
    ('irrelevant_features', DropColumns(irrelevant_features)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=True, funder=True)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital()),
    ('interactions', Interactions()),
    ('new_features', NewFeatures(num_private=False)),
    ('drop', DropColumns(drop_features))
])


# In[63]:


train_prep = transformation_pipeline.fit_transform(original_train)


# # 3. Feature Selection

# # 3.1 Information Gain

# In[64]:


def information_gain(df, feature, target):
    p_target = df[target].value_counts(normalize=True)
    H_target = np.nansum(-np.log2(p_target) * p_target)
    
    p_feature_target = df.groupby(feature)[target].value_counts(normalize=True)
    p_features = df[feature].value_counts(normalize=True)
    
    H_feature = 0
    for value in df[feature].unique():
        H_value = p_features[value] * np.nansum(-np.log2(p_feature_target[value]) * p_feature_target[value])
        H_feature += H_value
    
    return H_target - H_feature 


# In[65]:


features = train_prep.columns.drop('status_group')
information_gain_features = {}
for feature in features:
    information_gain_features[feature] = information_gain(train_prep, feature, 'status_group')
sorted(information_gain_features.items(), key=lambda tup: -tup[1]) 


# # 3.2 Logistic Regression L1 Regression

# In[66]:


X, y = train_prep[train_prep.columns.drop("status_group")], train['status_group']

transformation_pipeline = Pipeline([
    ('count_encoder', CountEncoder(handle_unknown=0)),
    ('scaler', StandardScaler())
])

X = transformation_pipeline.fit_transform(X)


# In[67]:


X, y = train_prep[train_prep.columns.drop("status_group")], train['status_group']

transformation_pipeline = Pipeline([
    ('count_encoder', CountEncoder(handle_unknown=0)),
    ('scaler', StandardScaler())
])

X = transformation_pipeline.fit_transform(X)

logistic = LogisticRegressionCV(Cs=[3, 10, 30, 100], penalty="l1", 
                                multi_class='multinomial',solver='saga', cv=3, n_jobs=-1).fit(X, y);
model = SelectFromModel(logistic, prefit=True);

X_new = model.transform(X);

selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=train_prep.index,
                                 columns=train_prep.drop('status_group', axis=1).columns)

dropped_columns = selected_features.columns[selected_features.var() == 0]
dropped_columns


# Lasso selection didn't remove any features, so we can continue with modeling and hyperparameter tuning

# # 4. Modeling and Evaluation

# # Split data to 80:20 ratio, and perform Model Selection

# In[136]:


from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

smote = SMOTE()

X_train, y_train = smote.fit_resample(X_train, y_train)
print(Counter(y_train))


# In[82]:


def runModel(train, test, target_train, target_test, model): 
    """Returns predicted values for training and testing set, along with the model's accruacy for both."""
    
    # Initialize and fit classification model
    mdl = model
    mdl.fit(train, target_train)    
    predTrain = mdl.predict(train)
    predTest = mdl.predict(test)
    
    trainAcc = accuracy_score(target_train, predTrain)
    testAcc = accuracy_score(target_test, predTest)
    
    return predTrain, predTest, trainAcc, testAcc


# In[140]:


def printStats(trainAcc, testAcc, wp_target_test, predTest):
    """Prints  results and confusion matrix."""
    print('The training set accuracy is equal to ' + str(round(trainAcc*100,2)) + '%.')
    print('The testing set accuracy is equal to ' + str(round(testAcc*100,2)) + '%.')
    print('The difference is ' + str(round((trainAcc-testAcc)*100,2)) + '%.\n')
    
    print(classification_report(wp_target_test, predTest))
    cm = confusion_matrix(wp_target_test, predTest)
    
    fig, ax = plt.subplots(figsize=(5.75, 5.75))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - Test', fontsize=18)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
    ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
    plt.show()
       
    return


# In[141]:



# to give model baseline report in dataframe 
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
    
    y_pred = model.predict(X_test)
   
    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'timetaken'    : [0]       })   # timetaken: to be used for comparison later
    return df_model


# In[17]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False))
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)


# # 4.1 Random Forest

# # 4.1.1 Random Forest without PCA

# In[15]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
    ('encoder', CountEncoder(handle_unknown=0))
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)

rf = RandomForestClassifier(n_jobs=-1)

# setting grid
param_grid_forest = {
    'max_depth': [10, 25, 50, None],
    'n_estimators': [50, 100]
}


# In[37]:


X_prep.info()


# In[45]:


grid_search_forest = GridSearchCV(rf, cv=5, param_grid=param_grid_forest,n_jobs=-1).fit(X_prep, y)


# In[152]:


print('Train performance on Random Forest w/o PCA: ', grid_search_forest.best_estimator_.score(X_prep, y))


# In[153]:


params = ['rank_test_score','mean_test_score', 'param_max_depth', 'param_n_estimators']
pd.DataFrame(grid_search_forest.cv_results_)[params].sort_values('rank_test_score')


# In[30]:


# Plot the top features based on its importance
(pd.Series(grid_search_forest.best_estimator_.feature_importances_, index=X_train.columns)
    .nlargest(10)   # can adjust based on how many top features you want
    .plot(kind='barh', figsize=[8,4])
    .invert_yaxis()) # Ensures that the feature with the most importance is on top, in descending order

plt.yticks(size=15)
plt.title('Top Features derived by Random Forest', size=20)


# In[17]:


sorted([*zip(X_prep.columns, grid_search_forest.best_estimator_.feature_importances_)], key=lambda tup: -tup[1])


# In[92]:


importances = grid_search_forest.best_estimator_.feature_importances_
#
# Sort the feature importance in descending order
#
plt.subplots(figsize=(12,10))
sorted_indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_prep.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_prep.shape[1]), X_prep.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()


# In[156]:


y_pred_cv = cross_val_predict(grid_search_forest.best_estimator_, X_prep, y, cv=5)
acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(5.75, 5.75))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[19]:


test.info()


# In[157]:


test_prep = transformation_pipeline.transform(test)
y_pred = grid_search_forest.best_estimator_.predict(test_prep)

test_id = pd.read_csv(path + 'Test_set.csv').id
submission = pd.DataFrame({"id": test_id, "status_group": y_pred})
submission = submission.replace({'status_group': num_to_class})
submission.to_csv(output_path + 'submission_rf.csv', index=False)


# In[18]:


X_prep.to_csv(output_path + 'tuned_data.csv', index=False)


# Notes:
# Our model is overfitting based on the difference between train performance (0.99) and cross-validation performance (0.81). Feature importance shows that wee need to drop two features "dry_season" and "water_quality". After removing those two features and reducing max_depth to 100, our generalization didn't improve. We figured out also that using "Eucledian" distance gives slightly better performance than "Manhattan" in transformer "Distance()".
# 
# Based on error analysis, we can see that Random Forest is doing a lot of mistakes in predicting class 1 which is "non functional needs repair". This is understable given the heavy imbalance of this class (around 7% of classes).
# 
# The best parameters based on CV:
# 
# n_estimators = 100
# max_depth = 25

# # 4.1.2 Random Forest with PCA

# In[32]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
    ('encoder', CountEncoder(handle_unknown=0)),
    ('scaler', StandardScaler()),
    ('boxcos', PowerTransformer()),
    ('PCA', PCA())
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)


# In[34]:




rf_pca = RandomForestClassifier(n_jobs=-1)

# setting grid
param_grid_forest = {
    'max_depth': [10, 25, 50,100, None],
    'n_estimators': [50, 100]
}

grid_search_forest_pca = GridSearchCV(rf_pca, cv=5, param_grid=param_grid_forest,n_jobs=-1).fit(X_prep, y)


# In[36]:


y_pred_cv = cross_val_predict(grid_search_forest_pca.best_estimator_, X_prep, y, cv=5)
acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(5.75, 5.75))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[38]:


print('Train performance on Random Forest w/ PCA: ', grid_search_forest_pca.best_estimator_.score(X_prep, y))


# In[98]:


params = ['rank_test_score','mean_test_score', 'param_max_depth', 'param_n_estimators']
pd.DataFrame(grid_search_forest_pca.cv_results_)[params].sort_values('rank_test_score')


# In[24]:


y_pred_cv = cross_val_predict(grid_search_forest_pca.best_estimator_, X_prep, y, cv=5)
conf_mx = confusion_matrix(y, y_pred_cv)
row_sums = conf_mx.sum(axis=1, keepdims=True) 
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[48]:


test_prep = transformation_pipeline.transform(test)
y_pred = grid_search_forest_pca.best_estimator_.predict(test_prep)

test_id = pd.read_csv(path + 'Test_set.csv').id
submission = pd.DataFrame({"id": test_id, "status_group": y_pred})
submission = submission.replace({'status_group': num_to_class})
submission.to_csv(output_path + 'submission_rf_pca.csv', index=False)


# Notes:
# Similar issues with Random Forest run without PCA, except for the performance on cross-validation got worse (from 0.81 to 0.77).

# # 4.2 XGBoost

# In[39]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
     ('encoder', CountEncoder(handle_unknown=0)),
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)


# In[40]:



xgb_model = XGBClassifier(n_jobs=-1)

# setting grid
#param_grid_xgb = {
    #'max_depth': [20],
    #'learning_rate': [0.03, 0.1, 0.3]
#}
# setting grid
eta = list(map(lambda x: x/100, list(range(9,20))))
subsample = [0.5,0.7,1]
#subsample = list(map(lambda x: x/10, list(range(5,10))))
#colsample = list(map(lambda x: x/10, list(range(5,10))))
colsample = [0.5,0.7,1]



param_grid_xgb = {
    'max_depth': [5,10,20],
    'learning_rate': [0.1, 0.2]
    
    
}

grid_search_xgb = GridSearchCV(xgb_model, cv=5, param_grid=param_grid_xgb, n_jobs=-1).fit(X_prep, y)


# In[41]:


print('Train performance on XGB: ', grid_search_xgb.best_estimator_.score(X_prep, y))


# In[164]:


params = ['rank_test_score','mean_test_score', 'param_max_depth', 'param_learning_rate']
pd.DataFrame(grid_search_xgb.cv_results_)[params].sort_values('rank_test_score')


# In[42]:


y_pred_cv = cross_val_predict(grid_search_xgb.best_estimator_, X_prep, y, cv=5)
acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(5.75, 5.75))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[ ]:


test_prep = transformation_pipeline.transform(test)
y_pred = grid_search_xgb.best_estimator_.predict(test_prep)

test_id = pd.read_csv(path + 'Test_set.csv').id
submission = pd.DataFrame({"id": test_id, "status_group": y_pred})
submission = submission.replace({'status_group': num_to_class})
submission.to_csv(output_path + 'submission_gbm_up.csv', index=False)


# Here we can note that the performance of the XGBoost classifier was marginally worse than that of the random forest classifier.

# # LDA Trial

# In[167]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
    ('encoder', CountEncoder(handle_unknown=0)),
    ('scaler', StandardScaler()),
    ('boxcos', PowerTransformer())
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)

lda = LDA()

# setting grid
param_grid_lda = {
    'n_components': [2, 10]
}

grid_search_forest_LDA = GridSearchCV(lda, cv=5, param_grid=param_grid_lda,n_jobs=-1).fit(X_prep, y)


# In[168]:


print('Train performance on LDA: ', grid_search_forest_LDA.best_estimator_.score(X_prep, y))


# In[169]:


params = ['rank_test_score','mean_test_score']
pd.DataFrame(grid_search_forest_LDA.cv_results_)[params].sort_values('rank_test_score')


# In[47]:


corr = pd.concat([pd.Series(grid_search_forest.best_estimator_.predict(test_prep), name="RF"),
                              
                              pd.Series(grid_search_lgb.best_estimator_.predict(test_prep), name="GBM"),
                              pd.Series(grid_search_xgb.best_estimator_.predict(test_prep), name="XGB")],axis=1)

plt.figure(figsize=(10,8))
sns.heatmap(corr.corr(),annot=True)
plt.show()


# The performance of the LDA classifier is significantly worse than those of the other classifiers used, and as such will not be further explored.

# # LightGBM

# In[20]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
    ('encoder', CountEncoder(handle_unknown=0))
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)

lgb = LGBMClassifier(objective='multiclass',num_class=3)

param_grid_lgb = {
    'num_leaves': [64,128],
    'learning_rate': [0.3, 1],
    'max_depth': [16],
    'min_data_in_leaf': [200, 500]
}

grid_search_lgb = GridSearchCV(lgb, cv=5, param_grid=param_grid_lgb,n_jobs=-1).fit(X_prep, y)


# In[18]:


print('Train performance on LightGBM: ', grid_search_lgb.best_estimator_.score(X_prep, y))


# In[16]:


params = ['rank_test_score','mean_test_score', 'param_num_leaves', 'param_learning_rate','param_min_data_in_leaf']
pd.DataFrame(grid_search_lgb.cv_results_)[params].sort_values('rank_test_score')


# In[21]:


y_pred_cv = cross_val_predict(grid_search_lgb.best_estimator_, X_prep, y, cv=5)
acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(5.75, 5.75))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[22]:


test_prep = transformation_pipeline.transform(test)
y_pred = grid_search_lgb.best_estimator_.predict(test_prep)

test_id = pd.read_csv(path + 'Test_set.csv').id
submission = pd.DataFrame({"id": test_id, "status_group": y_pred})
submission = submission.replace({'status_group': num_to_class})
submission.to_csv(output_path + 'submission_lgb_up.csv', index=False)


# In[88]:


x_prep.to_csv(output_path + 'tuneddata.csv', index=False)


# Notes:
# Best parameters based on CV:
# 
# num_leaves: 128
# learning_rate: 0.3
# min_data_in_leaf: 200
# max_depth: 16

# LightGBM showed significant improvement in performance, in terms of speed, when running for this model. The result of this classifier was marginally worse than that of the original classifier.

# # Voting Classifier

# Having ran the individual classifiers, combining the results obtained from them, may yield in a better overall model. As such, a voting classifier will be designed in which the best three individual models will be incorporated.

# In[15]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
    ('encoder', CountEncoder(handle_unknown=0))
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)

lgb_clf = LGBMClassifier(objective='multiclass',num_class=3, min_data_in_leaf=200, num_leaves=128, learning_rate=0.3, max_depth=16)
xgb_clf = XGBClassifier(n_jobs=-1, max_depth=10, learning_rate=0.2)
rf_clf = RandomForestClassifier(max_depth=25, n_estimators=100, n_jobs=-1)

voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf), 
    ('lgb', lgb_clf),
    ('xgb', xgb_clf)], voting='soft')

voting_clf.fit(X_prep, y)

voting_clf_hard = VotingClassifier(estimators=[
    ('rf', rf_clf), 
    ('lgb', lgb_clf),
    ('xgb', xgb_clf)], voting='hard')

voting_clf_hard.fit(X_prep, y)


# # Concatenated confusion matrix

# In[51]:


meaningless_cols = ['id','waterpoint_type_group', 'source_type', 'source_class', 
                   'quantity_group', 'quality_group', 'payment_type', 
                   'management_group', 'extraction_type_group', 'extraction_type_class', 
                   'scheme_name', 'recorded_by', 'district_code', 
                   'region_code', 'region','subvillage', 'public_meeting']

features_to_drop = ['latitude', 'longitude', 'date_recorded', 'num_private', 'permit', 'water_quality']

transformation_pipeline = Pipeline([
    ('meaningless_features', DropColumns(meaningless_cols)),
    ('simple_imputer', OurSimpleImputer(permit=False)),
    ('government', Govtcorrection(installer=False, funder=False)),
    ('geo_clusters', GeoClustering()),
    ('advanced_imputer', OurAdvancedImputer(population_bucket=False)),
    ('distance', Distance_from_capital(strategy='eucledian')),
    ('interactions', Interactions()),
    ('New_features', NewFeatures(num_private=False, dry_season=False)),
    ('drop', DropColumns(features_to_drop)),
    ('encoder', CountEncoder(handle_unknown=0))
])

class_to_num = {'functional': 2, 'non functional': 0, 'functional needs repair': 1}
num_to_class = {0:'non functional', 1: 'functional needs repair', 2: 'functional'} 

X = original_train.drop('status_group', axis=1)
y = original_train.status_group.replace(class_to_num)

X_prep = transformation_pipeline.fit_transform(X)

lgb_clf = LGBMClassifier(objective='multiclass',num_class=3, min_data_in_leaf=200, num_leaves=128, learning_rate=0.3, max_depth=16)
xgb_clf = XGBClassifier(n_jobs=-1, max_depth=10, learning_rate=0.2)
rf_clf = RandomForestClassifier(max_depth=25, n_estimators=100, n_jobs=-1)

voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf), 
    ('lgb', lgb_clf),
    ('xgb', xgb_clf)], voting='soft')


# In[27]:


X_prep.isnull().sum()


# In[63]:


def evaluate_model(data_x, data_y):
    #k_fold = KFold(5, shuffle=True, random_state=1)
   
    predicted_targets = np.array([])
    actual_targets = np.array([])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    lst_accu_stratified = []
    for train_ix, test_ix in skf.split(data_x, data_y):
        train_x, train_y, test_x, test_y = data_x.iloc[train_ix], data_y.iloc[train_ix], data_x.iloc[test_ix], data_y.iloc[test_ix]

        # Fit the classifier
        classifier = voting_clf.fit(train_x, train_y)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)
        lst_accu_stratified.append(voting_clf.score(test_x, test_y))
        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets


# In[64]:


def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()


# In[65]:


class_names = ['functional','functional needs repair', 'non functional']
def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


# In[66]:


predicted_target, actual_target = evaluate_model(X_prep, y)


# In[50]:


plot_confusion_matrix(predicted_target, actual_target)


# In[27]:


print('Train performance on Voting: ', voting_clf.score(X_prep, y))


# In[67]:


#y_pred_cv = cross_val_predict(voting_clf, X_prep, y, cv=5)
print(classification_report(predicted_target, actual_target))
cm = confusion_matrix(predicted_target, actual_target)
target_names = ['functional','functional needs repair', 'non functional']
cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# In[86]:


#y_pred_cv = cross_val_predict(voting_clf, X_prep, y, cv=5)
#acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
#print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(12, 10))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[79]:


y_pred_cv = cross_val_predict(voting_clf, X_prep, y, cv=5)
acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(5.75, 5.75))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[ ]:


ax = sns.heatmap(cm, annot=True, fmt='g')
plt.subplots(figsize=(12,10))
ax.set_title('Seaborn Confusion Matrix with labels!!');
ax.set_xlabel('Predicted Fruits')
ax.set_ylabel('Actual Fruits');
ax.xaxis.set_ticklabels(['apples', 'oranges', 'pears'])
ax.yaxis.set_ticklabels(['apples', 'oranges', 'pears'])
plt.show()


# In[80]:


#y_pred_cv = cross_val_predict(voting_clf_hard, X_prep, y, cv=5)
#acc = str(round(accuracy_score(y, y_pred_cv)*100,2)) + '%.'
#print("The Testing Accuracy score is : " + acc)
print(classification_report(y, y_pred_cv))
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(5.75, 5.75))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Test', fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['functional','functional needs repair', 'non functional'])
ax.set_yticklabels(['functional','functional needs repair', 'non functional'])
plt.show()


# In[ ]:


#group_names = ['True Neg','False Pos','False Neg','True Pos','True Pos','True Pos','True Pos','True Pos','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

labels = np.asarray(labels).reshape(3,3)

ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Flower Category')
ax.set_ylabel('Actual Flower Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
ax.yaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[29]:


print('Train performance on Voting: ', voting_clf_hard.score(X_prep, y))


# In[24]:


test_prep = transformation_pipeline.transform(test)
y_pred = voting_clf.predict(test_prep)

test_id = pd.read_csv(path + 'Test_set.csv').id
submission = pd.DataFrame({"id": test_id, "status_group": y_pred})
submission = submission.replace({'status_group': num_to_class})
submission.to_csv(output_path + 'submission_voting_soft.csv', index=False)


# In[25]:


test_prep = transformation_pipeline.transform(test)
y_pred = voting_clf_hard.predict(test_prep)

test_id = pd.read_csv(path + 'Test_set.csv').id
submission = pd.DataFrame({"id": test_id, "status_group": y_pred})
submission = submission.replace({'status_group': num_to_class})
submission.to_csv(output_path + 'submission_voting_hard.csv', index=False)


# In[42]:


test_prep = transformation_pipeline.transform(test)

test_prep.to_csv(output_path + 'test_prep.csv', index=False)


# This resulted in the best score through the submission on DrivenData, and as such is the final submission. The better result was recieved with the soft voting.
# 
# Remarks:
# There's a heavy class imbalance in the dataset, which results in one of the classes being predicted worse than the other two (functional, non functional). Potentially, this could be improved by applying oversampling technique like SMOTE in order to balance functional needs repair class to improve the results of the model.
# 
# Another thing that could be done, is trying to engineer other useful features or build a separate model just to learn specifically functional needs repair class.
