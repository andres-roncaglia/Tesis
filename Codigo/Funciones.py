# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para realizar consultas a la base de datos
import urllib.parse
import requests

# Para guardar y cargar los modelos
import pickle

# Para graficos
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------- FUNCIONES ------------------------------------

# Funcion get_api_call()
# Para hacer llamadas a la API de series argentinas
# argumentos:
# - ids : nombre de la serie
# - **kwargs : Argumentos adicionales
# salida: URL de la serie de tiempo desde la API (str)

def get_api_call(ids, **kwargs):
    API_BASE_URL = "https://apis.datos.gob.ar/series/api/"
    kwargs["ids"] = ",".join(ids)
    return "{}{}?{}".format(API_BASE_URL, "series", urllib.parse.urlencode(kwargs))

# ------------------------------------------------------------------------------------

# Funcion interval_score()
# Calcula el Interval Score de un pronostico 
# argumentos:
# - obs : Valores observados
# - lower : Valores del extremo inferior de intervalo 
# - upper : Valores del extremo superior de intervalo 
# - alpha : Nivel de significación del intervalo
# salida: Interval Score (numeric)

def interval_score(obs, lower, upper, alpha):

    upper = upper.values
    lower = lower.values
    obs = obs.values

    # Ancho del intervalo
    W = upper - lower

    # Penalización por sobre-estimación
    O = 2/alpha * np.maximum(lower - obs, 0)

    # Penalización por sub-estimación
    U = 2/alpha * np.maximum(obs - upper, 0)

    # Interval Score
    score = np.average(W + O + U)

    return score


# ------------------------------------------------------------------------------------

# Funcion plot_forecast()
# Grafica los pronosticos
# argumentos:
# - data : Pandas dataframe con los valores observados. Con columnas 'ds' e 'y'
# - forecast : Pandas dataframe con los pronosticos. Con columnas 'ds', 'pred', 'lower' y 'upper'
# - color : Color de los pronosticos en el grafico (opcional)
# - label : Etiqueta de los pronosticos en el grafico (opcional)
# salida: matplotlib plot

def plot_forecast(data, forecast, color = 'red', label = 'Prediccion', xlabel = 'Año', ylabel = 'Y'):
    plt.plot(data['ds'], data['y'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.lineplot(x = forecast['ds'], y= forecast['pred'], color = color, label = label)
    plt.fill_between(forecast['ds'], forecast['lower'], forecast['upper'], color = color, alpha = 0.3)


# ------------------------------------------------------------------------------------

# Funcion is_pickleable()
# Verifica si una variable es pickeable 
# argumentos:
# - obj : Variable
# salida: True o False

def is_pickleable(obj):
    """Check if an object can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, TypeError):
        return False

# ------------------------------------------------------------------------------------

# Funcion save_env()
# Guarda todas las variables pickeables del ambiente
# argumentos:
# - filename : Nombre del archivo .pkl a generar
# salida: Nada, Crea un archivo .pkl

def save_env(env_dict=None, filename="environment.pkl"):
    if env_dict is None:
        env_dict = globals()

    global_vars = {
        k: v for k, v in env_dict.items()
        if not k.startswith("__") and is_pickleable(v)
    }

    with open(filename, "wb") as f:
        pickle.dump(global_vars, f)

# ------------------------------------------------------------------------------------

# Funcion load_env()
# Carga al ambiente todas las variables guardadas en un archivo .pkl
# argumentos:
# - filename : Nombre del archivo .pkl a cargar
# salida: Nada, Crea un archivo .pkl

def load_env(filename="environment.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------------

