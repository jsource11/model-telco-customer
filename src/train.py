# Código de Entrenamiento 
###########################

import pandas as pd
import xgboost as xgb
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')

    # Preprocessing
    # Replace values in the 'Churn' column
    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
   


    # Define numeric and categorical features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                            'Contract', 'PaperlessBilling', 'PaymentMethod']

    # Create preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()) 
    ])
    
    # Create preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  
    ])
    
    # Combine preprocessing steps for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Modeling
    
    # Define classifiers/models
    classifiers = [
        ('logreg', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier()),
        ('gbc', GradientBoostingClassifier())
    ]
    
    # Feature selection and Voting Classifier
    voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
    
    # Define features (X) and target variable (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']  
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline with SMOTE
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE()),
        ('voting_clf', voting_clf)
    ])
    
    param_grid = {
        'voting_clf__logreg__C': [0.1, 1.0, 10],
        'voting_clf__rf__n_estimators': [50, 100, 200],
        'voting_clf__gbc__learning_rate': [0.01, 0.1, 0.2]
    }
    
    # Entrenamos el modelo con toda la muestra
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    print('Modelo entrenado')

    
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(grid_search, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('telco_customer_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()