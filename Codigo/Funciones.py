# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para guardar y cargar los modelos
import pickle

# Para graficos
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------- FUNCIONES ------------------------------------

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

def plot_forecast(data, forecast, pred_color = 'red', line_color = 'black', label = 'Prediccion', xlabel = 'Año', ylabel = 'Y', long=None):
    if long != None:
        data_plt = data.copy()
        data_plt = data_plt.tail(long)
    else:
        data_plt = data

    plt.plot(data_plt['ds'], data_plt['y'], color=line_color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.lineplot(x = forecast['ds'], y= forecast['pred'], color = pred_color, label = label)
    plt.fill_between(forecast['ds'], forecast['lower'], forecast['upper'], color = pred_color, alpha = 0.3)


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

