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

# Para guardar y cargar los modelos
from joblib import dump, load

# Para medir el tiempo que tarda en ajustar los modelos
import time


# Creamos una función para realizar llamadas a la API de datos argentina
def get_api_call(ids, **kwargs):
    API_BASE_URL = "https://apis.datos.gob.ar/series/api/"
    kwargs["ids"] = ",".join(ids)
    return "{}{}?{}".format(API_BASE_URL, "series", urllib.parse.urlencode(kwargs))



# Llamada a la API y carga de datos

api_call = get_api_call(["364.3_LITORAL_GAGAS__11"], start_date="2016-01")

json = requests.get(api_call).json()

datos = pd.DataFrame(json['data'], columns = ['fecha', 'consumo'])

datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')

datos.columns = ['ds', 'y']

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

corte = len(datos)-12

datos_train = datos[:corte]
datos_test = datos[corte:]

datos_train['item_id'] = 0

datos_train.columns = ['timestamp', 'target', 'item_id']

datos_train = TimeSeriesDataFrame(datos_train)

predictor = TimeSeriesPredictor(prediction_length=12).fit(datos_train, presets="bolt_tiny")

predictions = predictor.predict(datos_train)