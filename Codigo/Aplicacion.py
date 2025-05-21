# ------------------------------- CARGA DE LIBRERIAS -------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para dar formato fecha
from datetime import datetime

# Para realizar consultas a la base de datos
import urllib.parse
import requests

# Para graficos
import matplotlib.pyplot as plt
import seaborn as sns

# Para medir el tiempo que tarda en ajustar los modelos
import time

# Cargamos funciones
from Codigo.Funciones import get_api_call, interval_score, plot_forecast, save_env, load_env
from Codigo.tuner_fun import Tuner

# Definimos una semilla
seed = 11072001

# Cargamos el ambiente
# globals().update(load_env('Ambiente/resultados.pkl'))

# -------------------------------------------------------------------------
# ------------------------------- SERIE 1 ---------------------------------
# -------------------------------------------------------------------------


# ------------------------------- 1.1 CARGA DE DATOS -------------------------------

atenciones_guardia = pd.read_excel(io='Datos/Atenciones de guardia en el HNVV por patologías respiratorias (vigiladas por epidemiología).xlsx' )

# Filtro las columnas importantes
atenciones_guardia = atenciones_guardia[['fec', 'frec']]

# Aseguro que la columna fecha tenga el formato adecuado
atenciones_guardia['fec'] = pd.to_datetime(atenciones_guardia['fec'], format='%Y-%m-%d')

# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2
long_pred = 12

# Creamos un dataframe donde vamos a guardar todos los resultados
metricas_1 = pd.DataFrame(columns=(['Modelo', 'MAPE', 'Interval Score', 'Tiempo']))

# ------------------------------- 1.2 ARIMA -------------------------------


# Definimos los parametros a tunear
params = {
    "start_p" : 0, "max_p" : 3,
    "start_q" : 0, "max_q" : 3,
    "start_P" : 0, "max_P" : 3,
    "start_Q" : 0, "max_Q" : 3,
    "max_d":2, "max_D": 2,
    "random_state" : seed
}

# Tuneamos los parametros y ajustamos el modelo
resultados_1_arima = Tuner(forecaster_fun= 'ARIMA', datos=atenciones_guardia, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['ARIMA', resultados_1_arima['mape'], resultados_1_arima['score'], resultados_1_arima['tiempo']]

# ------------------------------- 1.3 XGBOOST -------------------------------


# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150]

}

# Tuneamos los parametros y ajustamos el modelo
resultados_1_xgb = Tuner(forecaster_fun= 'XGBoost', datos=atenciones_guardia, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['XGBoost', resultados_1_xgb['mape'], resultados_1_xgb['score'], resultados_1_xgb['tiempo']]

# ------------------------------- 1.4 LIGHTGBM -------------------------------


# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [5, 10, 20, 30, 50]
}

# Tuneamos los parametros y ajustamos el modelo
resultados_1_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=atenciones_guardia, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LightGBM', resultados_1_lgbm['mape'], resultados_1_lgbm['score'], resultados_1_lgbm['tiempo']]


# ------------------------------- 1.5 LSTM -------------------------------

# Definimos los parametros a tunear
parametros = {
    'max_steps' : [50, 100, 200, 500],
    'random_seed' : [seed],
    'encoder_n_layers' : [1,2,3],
    'decoder_layers' : [1,2,3]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_1_lstm = Tuner(forecaster_fun= 'LSTM', datos=atenciones_guardia, parametros= parametros, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LSTM', resultados_1_lstm['mape'], resultados_1_lstm['score'], resultados_1_lstm['tiempo']]


# ------------------------------- 1.6 TIMEGPT -------------------------------


# Definimos los parametros a tunear
parametros = {
    'finetune_loss' : ['mape'],
    'finetune_steps' : [1,2,5,10,15],
    'finetune_depth' : [1, 2, 3, 5]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_1_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=atenciones_guardia, parametros= parametros, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['TimeGPT', resultados_1_gpt['mape'], resultados_1_gpt['score'], resultados_1_gpt['tiempo']]



# -------------------------------------------------------------------------
# ------------------------------- SERIE 2 ---------------------------------
# -------------------------------------------------------------------------


# ------------------------------- 2.1 CARGA DE DATOS -------------------------------


# ------------------------------- 2.2 ARIMA -------------------------------


# ------------------------------- 2.3 XGBOOST -------------------------------


# ------------------------------- 2.4 LIGHTGBM -------------------------------


# ------------------------------- 2.5 LSTM -------------------------------


# ------------------------------- 2.6 TIMEGPT -------------------------------






# -------------------------------------------------------------------------
# ------------------------------- SERIE 3 ---------------------------------
# -------------------------------------------------------------------------


# ------------------------------- 3.1 CARGA DE DATOS -------------------------------


# ------------------------------- 3.2 ARIMA -------------------------------


# ------------------------------- 3.3 XGBOOST -------------------------------


# ------------------------------- 3.4 LIGHTGBM -------------------------------


# ------------------------------- 3.5 LSTM -------------------------------


# ------------------------------- 3.6 TIMEGPT -------------------------------





# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# Guardamos el ambiente
# save_env(env_dict=globals(), filename="Codigo/Ambiente/Amb_Aplicacion.pkl")