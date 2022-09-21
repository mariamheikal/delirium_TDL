import pandas as pd
import numpy as np
import datawig

#preparing dataset
def prepare(df):
    df=df[df.AGE>18]
    df_cols=list(df)
    if ['SUBJECT_ID','HADM_ID', 'ICUSTAY_ID'] in df_cols:
        df.drop(['SUBJECT_ID','HADM_ID', 'ICUSTAY_ID'],axis=1,inplace=True)
    cols_to_drop=[]
    cols=list(df)
    for col in df_cols:
        if df[col].isnull().sum()/_shape[0]>0.30:
            cols_to_drop.append(col)
    df.drop(cols_to_drop,inplace=True,axis=1)
    df.replace({'ETHNICITY': {'HISPANIC/LATINO - COLOMBIAN':'HISPANIC',
                                        'HISPANIC OR LATINO':'HISPANIC',
                                        'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)':'HISPANIC',
                                        'HISPANIC/LATINO - DOMINICAN':'HISPANIC',
                                        'HISPANIC/LATINO - CUBAN':'HISPANIC',
                                        'HISPANIC/LATINO - GUATEMALAN':'HISPANIC',
                                        'HISPANIC/LATINO - HONDURAN':'HISPANIC',
                                        'HISPANIC/LATINO - MEXICAN':'HISPANIC',
                                        'HISPANIC/LATINO - PUERTO RICAN':'HISPANIC',
                                        'HISPANIC/LATINO - SALVADORAN':'HISPANIC',
                                        'AMERICAN INDIAN/ALASKA NATIVE':'AMERICAN INDIAN',
                                        'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE':'AMERICAN INDIAN',
                                        'ASIAN - ASIAN INDIAN':'ASIAN', 
                                        'ASIAN - CAMBODIAN':'ASIAN',
                                        'ASIAN - CHINESE':'ASIAN', 
                                        'ASIAN - FILIPINO':'ASIAN', 
                                        'ASIAN - JAPANESE':'ASIAN',
                                        'ASIAN - KOREAN':'ASIAN', 
                                        'ASIAN - OTHER':'ASIAN', 
                                        'ASIAN - THAI':'ASIAN',
                                        'ASIAN - VIETNAMESE':'ASIAN',
                                        'BLACK/AFRICAN':'BLACK', 
                                        'BLACK/AFRICAN AMERICAN':'BLACK',
                                        'BLACK/CAPE VERDEAN':'BLACK', 
                                        'BLACK/HAITIAN':'BLACK',
                                        'WHITE - BRAZILIAN':'WHITE', 'WHITE - EASTERN EUROPEAN':'WHITE',
                                        'WHITE - OTHER EUROPEAN':'WHITE', 'WHITE - RUSSIAN':'WHITE',
                                        'PATIENT DECLINED TO ANSWER':'UNKNOWN/NOT SPECIFIED',
                                        'UNABLE TO OBTAIN':'UNKNOWN/NOT SPECIFIED',
                                       }}, inplace=True)
    return df
    

#handling missing values
def mergeDFs(df,df_imputed,imputed_col):
  l=imputed_col+'_imputed_proba'
  if l not in list(df_imputed):
    df_imputed2=df_imputed.drop(imputed_col,axis=1)
  else:
    df_imputed2=df_imputed.drop([imputed_col,l],axis=1)
  df=df[df[imputed_col].isnull()==False]
  df_imputed2=df_imputed2.rename(columns = {imputed_col+'_imputed':imputed_col})
  return pd.concat([df,df_imputed2])

def impute(df):
    cols=list(df)
    for col in cols:
        missing = df[df[col].isnull()]
        if missing.shape[0]>0:
            in_cols=cols
            in_cols.remove(col)
            imputer = datawig.SimpleImputer(
            input_columns=in_cols,
            output_column=col
            )
            imputer.fit(train_df = df)
            imputed = imputer.predict(missing)
            df=mergeDFs(df,imputed,col)
    return df
