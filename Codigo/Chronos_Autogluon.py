# ------------------------------------- LIBRERIAS -------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para dar formato fecha
from datetime import datetime

# Para graficos
import matplotlib.pyplot as plt

# Para realizar consultas a la base de datos
import urllib.parse
import requests

# Para calcular el error medio cuadratico
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# Funciones utiles
import sys
sys.path.append('Codigo')
from Funciones import interval_score, save_env, load_env, get_api_call, plot_forecast

# Para medir el tiempo que tarda en ajustar los modelos
import time

# ---------------------------------- CARGA DE DATOS ----------------------------------

# Cargamos el ambiente
globals().update(load_env('Codigo/Ambiente/resultados_chronos.pkl'))

# Opciones de pronostico

long_pred = 12
alpha = 0.2

# Llamada a la API y carga de datos

api_call = get_api_call(["364.3_LITORAL_GAGAS__11"], start_date="2016-01")
json = requests.get(api_call).json()
datos = pd.DataFrame(json['data'], columns = ['fecha', 'consumo'])
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')

# Renombramos las columnas de una forma que autogluon las pueda manejar

datos.columns = ['timestamp', 'target']

datos['item_id'] = 0

# Dividimos los datos que queremos pronosticar y con los que vamos a entrenar

corte = len(datos)-long_pred

datos_train = datos[:corte]
datos_test = datos[corte:]

# Esto nos sirve mas adelante para graficar
datos_b = datos.copy()
datos_test_b = datos_test.copy()

# ---------------------------------- AJUSTE DEL MODELO ----------------------------------

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Definimos una semilla
seed = 11072001

# Transformamos el dataset a TimeSeriesDataFrame
datos_train = TimeSeriesDataFrame(datos_train)

# Calculamos los cuantiles
q_lower = alpha/2
q_upper = 1-alpha/2

# Definimos y ajustamos el modelo
predictor = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
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
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape_chronos = mean_absolute_percentage_error(datos_test['target'], predictions['pred'])
score_chronos = interval_score(obs=datos_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos
datos_b.drop('item_id', axis = 1, inplace = True)
datos_test_b.drop('item_id', axis = 1, inplace = True)
datos_b.columns = ['ds', 'y']
datos_test_b.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
datos_test_b.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : datos_test_b['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Graficamos el pronostico
plot_forecast(data = datos_b, forecast = pred_chronos, color = 'violet', label = 'Chronos')
plt.show()

# Guardamos las metricas
salida = predictor.fit_summary()
resultados_chronos = salida['leaderboard']

# Guardamos los resultados
save_env(env_dict={
    'resultados_chronos' : resultados_chronos, 'datos_b': datos_b,
    'pred_chronos' : pred_chronos, 'mape_chronos' : mape_chronos, 
    'score_chronos' : score_chronos
    }, filename="Codigo/Ambiente/resultados_chronos.pkl")

# Crear una metrica propia
# https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html
