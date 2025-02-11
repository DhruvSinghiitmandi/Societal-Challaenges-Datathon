#!/usr/bin/env python

import os
import requests
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


logging.basicConfig(level=logging.INFO)

base_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"

def download_file(data_dir,file_name):
    url = base_url + file_name
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        logging.info(f"Downloaded {file_name} to {data_dir}")
    else:
        logging.error(f"Failed to download {file_name}")
def convert_xpt_to_csv(file_name):
    input_path = os.path.join('data', file_name)
    output_path = os.path.join('data', file_name.replace('.XPT', '.csv'))
    df = pd.read_sas(input_path)
    df.to_csv(output_path, index=False)
    logging.info(f"Converted {file_name} to {output_path}")


def download_data(data_dir):
    file_names = [
    "DEMO_L.XPT", "DIQ_L.XPT", "GLU_L.XPT", "ALQ_L.XPT", 
    "SMQ_L.XPT", "BMX_L.XPT", "BPQ_L.XPT", "MCQ_L.XPT", 
    "PAQ_L.XPT", "WHQ_L.XPT","SLQ_L.XPT","DR1TOT_L.XPT",
    "INQ_L.XPT","OCQ_L.XPT"
]
    for file_name in file_names:
        download_file(data_dir,file_name)
    for file_name in file_names:
        convert_xpt_to_csv(file_name)


def merge_data(data_dir):
    '''Import data'''
    logging.info('Loading XPT data...')
    DEMO = pd.read_csv(data_dir + '/DEMO_L.csv')
    DIQ = pd.read_csv(data_dir + '/DIQ_L.csv')
    GLU = pd.read_csv(data_dir + '/GLU_L.csv')
    ALQ = pd.read_csv(data_dir + '/ALQ_L.csv')
    SMQ = pd.read_csv(data_dir + '/SMQ_L.csv')
    BMX = pd.read_csv(data_dir + '/BMX_L.csv')
    BPQ = pd.read_csv(data_dir + '/BPQ_L.csv')
    MCQ = pd.read_csv(data_dir + '/MCQ_L.csv')
    PAQ = pd.read_csv(data_dir + '/PAQ_L.csv')
    WHQ = pd.read_csv(data_dir + '/WHQ_L.csv')
    DR1TOT = pd.read_csv(data_dir + '/DR1TOT_L.csv')
    INQ = pd.read_csv(data_dir + '/INQ_L.csv')
    SLQ = pd.read_csv( data_dir +  '/SLQ_L.csv')
    OCQ = pd.read_csv(data_dir + '/OCQ_L.csv')

    DEMO_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', ]
    DIQ_cols = ['SEQN', 'DIQ010']
    GLU_cols = ['SEQN', 'LBXGLU']
    ALQ_cols = ['SEQN', 'ALQ120Q']
    SMQ_cols = ['SEQN', 'SMQ020']
    BMX_cols = ['SEQN', 'BMXHT', 'BMXWAIST', 'BMXBMI']
    BPQ_cols = ['SEQN', 'BPQ020']
    MCQ_cols = ['SEQN','MCQ160M']
    PAQ_cols = ['SEQN', 'PAD790Q']
    DR1TOT_cols = ['SEQN','DR1TKCAL','DR1TCARB','DR1TSUGR','DR1TTFAT','DR1TCHOL']
    SLQ_cols = ['SEQN','SLD012']
    INQ_cols = ['SEQN','INDFMMPI','INDFMMPC','INQ300','IND310']
    OCQ_cols = ['SEQN','OCD150']

   
    logging.info('Merging data...')
    age = 0
    df_00 = DEMO[DEMO_cols] \
            .merge(DIQ[DIQ_cols], on='SEQN') \
            .merge(SMQ[SMQ_cols], on='SEQN') \
            .merge(BMX[BMX_cols], on='SEQN') \
            .merge(BPQ[BPQ_cols], on='SEQN') \
            .merge(MCQ[MCQ_cols], on='SEQN') \
            .merge(PAQ[PAQ_cols], on='SEQN') \
            .merge(SLQ[SLQ_cols], on='SEQN') \
            .merge(DR1TOT[DR1TOT_cols], on='SEQN') \
            .merge(OCQ[OCQ_cols] , on='SEQN') \
            .merge(GLU[GLU_cols], on='SEQN', how='left')
            # .merge(GLU[GLU_cols], on='SEQN') \
    # .merge(DR1TOT[DR1TOT_cols], on='SEQN') \
            # .merge(GLU[GLU_cols], on='SEQN') \ 
    print(df_00.shape)

    import matplotlib.pyplot as plt

    # Create a temporary DataFrame to mirror the final status calculation
    temp_df = df_00.copy()
    temp_df.loc[~temp_df['LBXGLU'].isna(), 'DIQ010'] = 2 - (temp_df['LBXGLU'] >= 126).astype(int)
    temp_df['status'] = temp_df['DIQ010'].apply(lambda x: 1 if x == 1 else 0)

    # Count occurrences: 1 is Diabetic, anything else is Non Diabetic
    counts = temp_df['status'].value_counts()
    non_diabetic = counts.get(0, 0)
    diabetic = counts.get(1, 0)

 
    
    df_00.loc[~df_00['LBXGLU'].isna(), 'DIQ010'] = 2 - (df_00['LBXGLU'] >= 126).astype(int)

    df_pop = pd.concat([df_00])



    # df = pd.concat([diag_total, undiag_total, prediab_total, nodiab_total], ignore_index=True)
    df = df_00.copy()
    df = df.drop(['SEQN','LBXGLU'], axis=1)

    import matplotlib.pyplot as plt

    # Count the values in the status column
    
    rename_dict = {
    'DIQ010': 'status',
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender',
    'RIDRETH3': 'Race',
    'WTINT2YR': 'Weight',
    'DMDEDUC2': 'Education',
    'ALQ151': 'AlcoholUse',
    'BMXHT': 'Height',
    'BMXWAIST': 'WaistCircumference',
    'BMXBMI': 'BMI',
    'BPQ020': 'Hypertension',
    'PAD790Q': 'PhysicalActivity',
    'DR1TKCAL': 'Calories',
    'DR1TCARB': 'Carbs',
    'DR1TTFAT': 'TotalFat',
    'DR1TCHOL': 'Cholesterol',
    'DR1TSUGR': 'Sugar',
    'SLD012': 'SleepDuration',
    'MCQ160M': 'Thyroid',
    'MCQ510D': 'viral hepatitis',
    'SMQ020': 'Smoked at least 100 cigarettes in life'

}
    

# Rename columns using rename_dict
    df = df.rename(columns=rename_dict)
    df['status'] = df['status'].apply(lambda x: 1.0 if x == 3.0 else x)
    df['Education'] = df['Education'].replace(9, np.nan)
    df['OCD150'] = df['OCD150'].replace(9, np.nan)
    df['Smoked at least 100 cigarettes in life'] = df['Smoked at least 100 cigarettes in life'].replace(9, np.nan)
    df = df[~df['status'].isin([7.0, 9.0])]

    df['status'] = df['status'].apply(lambda x: 1.0 if x == 1.0 else 0)
    df['PhysicalActivity'] = df['PhysicalActivity'].replace([7777, 9999], np.nan)
    df['Thyroid'] = df['Thyroid'].replace([7, 9], np.nan)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=289)


    import matplotlib.pyplot as plt
    counts = df['status'].value_counts()
    diabetic = counts.get(1, 0)
    non_diabetic = counts.get(0, 0)


    '''Save data'''
    fname_train = os.path.join(data_dir, 'diabetes_data_train.csv')
    fname_test = os.path.join(data_dir, 'diabetes_data_test.csv')
    df_train.to_csv(fname_train, index=False, float_format='%.1f')
    logging.info('Training set saved: {}'.format(fname_train))
    df_test.to_csv(fname_test, index=False, float_format='%.1f')
    logging.info('Test set saved: {}'.format(fname_test))


if __name__ == '__main__':
    data_dir = 'data'
    # download_data(data_dir)
    merge_data(data_dir)
