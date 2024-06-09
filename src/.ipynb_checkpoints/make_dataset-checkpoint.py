# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os


# Leemos los archivos csv
def read_file_csv(filename):
    # Display all columns and rows in the output
    pd.set_option("display.max_columns",None)
    pd.set_option('display.max_rows',None)
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    # i will drop this column because it isn't useful
    df.drop(columns=['customerID'], inplace=True)
    
    # Convert 'TotalCharges' to numeric, setting errors='coerce' to handle spaces and non-numeric values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Columns suitable for conversion to categorical
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]
    # Convert each column to categorical
    for col in categorical_cols:
        df[col] = df[col].astype('category')


    # Drop full duplicates
    df.drop_duplicates(inplace=True)

    nans = df.isna().sum().sort_values(ascending=False)
    pct = 100 * nans / df.shape[0]
    nan_stats = pd.concat([nans, pct], axis=1)
    nan_stats.columns = ['num_of_nans', 'percentage_of_nans']

    df.dropna(subset=['TotalCharges'], inplace=True)


    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn'],'telco_customer_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('WA_Fn-UseC_-Telco-Customer-Churn-new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn'],'telco_customer_val.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('WA_Fn-UseC_-Telco-Customer-Churn-new-score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn'],'telco_customer_score.csv')
    
    
if __name__ == "__main__":
    main()