# ------------------------------------- LIBRERIAS -------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para dar formato fecha
from datetime import datetime

# Para graficos
import matplotlib.pyplot as plt
import seaborn as sns

# Para realizar consultas a la base de datos
import urllib.parse
import requests

# Para calcular el error medio cuadratico
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# Funciones utiles
from Funciones import interval_score, save_env, load_env, get_api_call

# Para medir el tiempo que tarda en ajustar los modelos
import time

# ---------------------------------- CARGA DE DATOS ----------------------------------

# Llamada a la API y carga de datos

api_call = get_api_call(["364.3_LITORAL_GAGAS__11"], start_date="2016-01")

json = requests.get(api_call).json()

datos = pd.DataFrame(json['data'], columns = ['fecha', 'consumo'])

datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')

# Renombramos las columnas de una forma que autogluon las pueda manejar

datos.columns = ['timestamp', 'target']

datos['item_id'] = 0

# Dividimos los datos que queremos pronosticar y con los que vamos a entrenar

corte = len(datos)-12

datos_train = datos[:corte]
datos_test = datos[corte:]

# ---------------------------------- AJUSTE DEL MODELO ----------------------------------

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Definimos una semilla
seed = 11072001

# Transformamos el dataset a TimeSeriesDataFrame
datos_train = TimeSeriesDataFrame(datos_train)

# Definimos y ajustamos el modelo
predictor = TimeSeriesPredictor(
    
    prediction_length=len(datos_test),
    quantile_levels =  [0.1, 0.9],
    eval_metric = 'MAPE',
    ).fit(
    datos_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_tiny", "ag_args": {"name_suffix": "tiny-ZeroShot"}},
            {"model_path": "bolt_tiny", "fine_tune": True, "ag_args": {"name_suffix": "tiny-FineTuned"}},
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "small-ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "small-FineTuned"}},
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = predictor.predict(datos_train)

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(datos_test['y'], predictions['mean'])



# Crear una metrica propia
# https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html
