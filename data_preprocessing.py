import pandas as pd
import numpy as np
import datawig
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import quantile_transform
from sklearn.model_selection import train_test_split
#preparing dataset
def prepare(df,threshold,cols_to_drop):
    df=df[df.AGE>18]
    df_cols=list(df)
    cols_to_drop=[]
    _shape=df.shape
    for col in df_cols:
        if df[col].isnull().sum()/_shape[0]>threshold:
            cols_to_drop.append(col)
    df.drop(cols_to_drop,inplace=True,axis=1)
    df.drop_duplicates(inplace=True)
    #cols_to_drop=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',"Unnamed: 0"]
    for col in cols_to_drop:
        if col in list(df):
            df.drop(col,inplace=True,axis=1)
    return df

def downsample(df,target,factor):
  target_labels=df[target].values
  df = df.sample(frac=1)
  cols=list(df)
  X=df.values
  onesX=[]
  zeroesX=[]
  for x in X:
    if x[-1]==1:
      onesX.append(x)
    else:
      zeroesX.append(x)
  zer_len=int(len(onesX)*factor)
  X=onesX+zeroesX[0:zer_len]
  df_downsampled=pd.DataFrame(X,columns=cols)
  return df_downsampled

def unique(lst):
    lst_to_set = set(lst)
    unique_list = list(lst_to_set)
    return unique_list

def find_categorical_cols(df,target="",cat_col_thresh):
  cols=df.columns
  cat_features=[]
  i=0
  for col in cols:
      if len(unique(df[col]))<=cat_col_thresh and col!=target:
          cat_features.append(col)
          i=i+1
  return cat_features

#handling missing values
def mergeDFs(df,df_imputed,imputed_col):
  l=imputed_col+'_imputed_proba'
  if l not in list(df_imputed):
    df_imputed2=df_imputed.drop(imputed_col,axis=1)
  else:
    df_imputed2=df_imputed.drop([imputed_col,l],axis=1)
  df=df[df[imputed_col].isnull()==False]
  df_imputed2=df_imputed2.rename(columns = {imputed_col+'_imputed':imputed_col})
  return pd.concat([df,df_imputed2],sort=False)

def impute(df_train,df_test):
    cols=list(df_train)
    for col in cols:
        missing = df_train[df_train[col].isnull()]
        missing_test=df_test[df_test[col].isnull()]
        if missing.shape[0]>0 or missing_test.shape[0]>0:
            in_cols=cols
            in_cols.remove(col)
            imputer = datawig.SimpleImputer(
            input_columns=in_cols,
            output_column=col
            )
            imputer.fit(train_df = df_train)
            imputed = imputer.predict(missing)
            imputed_test = imputer.predict(missing_test)
            df_train=mergeDFs(df_train,imputed,col)
            df_test=mergeDFs(df_test,imputed_test,col)
    for col in cols:
        mvals=df_train[col].isnull().sum()
        mvals_test=df_test[col].isnull().sum()
        if mvals>0 or mvals_test>0:
            df_train, df_test=impute(df_train,df_test)
    return df_train, df_test


def encode(df_train,df_test,target,cat_col_thresh):
  
  cat_cols=find_categorical_cols(df_train,target,cat_col_thresh)
  y = df_train[target].values
  X=df_train.drop([target], axis=1)
  ytest = df_test[target].values
  Xtest=df_test.drop([target], axis=1)
  encoder = ce.LeaveOneOutEncoder()
  train_looe = encoder.fit_transform(X[cat_cols], y)
  test_looe = encoder.transform(Xtest[cat_cols])    

  cols=list(df_train)
  cols2=list(train_looe)
  for col in cols:
      if col not in cols2:
          train_looe[col]=df_train[col]
  train_looe[target]=y
  cols=list(df_test)
  cols2=list(test_looe)
  for col in cols:
      if col not in cols2:
          test_looe[col]=df_test[col]
  test_looe[target]=ytest
  return train_looe,test_looe


def normalize(df_train,df_test,target,scale_target=False):
  X_train=df_train.drop(target,axis=1).values
  X_train =quantile_transform(X_train, random_state=0, copy=True)
  cols=list(df_train)
  cols.remove(target)
  df_train_scaled=pd.DataFrame(X_train,columns=cols)
  df_train_scaled[target]=df_train[target]
  X_test = df_test.drop(target,axis=1).values
  X_test = quantile_transform(X_test, random_state=0, copy=True)
  cols=list(df_test)
  cols.remove(target)
  df_test_scaled=pd.DataFrame(X_test,columns=cols)
  df_test_scaled[target]=df_test[target]
  if scale_target:
    scale = StandardScaler()
    X = df_train_scaled[[target]].values
    Xts=df_test_scaled[[target]].values
    scaledX = scale.fit_transform(X)
    scaledXts = scale.transform(Xts)
    df_train_scaled[target]=scaledX
    df_test_scaled[target]=scaledXts
  return df_train_scaled,df_test_scaled

def split(df,target):
  df=df.sample(frac=1, random_state=0)
  y = df[target].values
  X=df.drop([target], axis=1)
  xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=0)
  xtrain[target]=ytrain
  xtest[target]=ytest
  return xtrain, xtest

#scale target option
def pipeline(df, target, prepare_f=False, split_f=False,
impute_f=False, normalize_f=False, downsample_f=False, 
encode_f=False, scale_target_f=False, threshold=0.35, factor=2,cols_to_drop=[],cat_col_thresh):
  #downsampling data
  if downsample_f:
    df=downsample(df,target,factor)
    
  #preporcess data
  if prepare_f:
    df=prepare(df,threshold,cols_to_drop)

  #split data
  if split_f:
    df_train,df_test=split(df,target)

  #impute missing values
  if impute_f:
    df_train,df_test=impute(df_train,df_test)

  #encode categorical variables
  if encode_f:
    df_train,df_test=encode(df_train,df_test,target,cat_col_thresh)

  #normalize numerical data
  if normalize_f:
    df_train,df_test=normalize(df_train,df_test,scale_target_f)

  return df_train,df_test

