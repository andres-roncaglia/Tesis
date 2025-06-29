# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para guardar y cargar los modelos
import pickle

# Para graficos
import matplotlib.pyplot as plt
import seaborn as sns

# Para autocorrelaciones
from statsmodels.tsa.stattools import acf, pacf

# Para calcular el test de ljung box
import statsmodels.api as sm

# Para la transformacion de box y cox y el test de normalidad
from scipy import stats

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
# Funcion autocorr_plot()
# Calcula y grafica las autorrelaciones y autocorrelaciones parciales


def autocorr_plot(data, lags, atype = 'acf'):
    vector = data.copy().dropna()

    if atype == 'acf':
        autocorr = acf(vector, nlags=lags)
        lags = np.arange(lags+1)
    else:
        autocorr = pacf(vector, nlags=lags, method = 'ywm')
        lags = np.arange(lags+1)

        autocorr = autocorr[1:]
        lags = lags[1:]

    # Calculo las autocorrelaciones
    autocorrelaciones = pd.DataFrame({
        'lag' : lags,
        atype : autocorr
    })

    # Defino las bandas limite
    upper_bound = 1.96/np.sqrt(len(vector))
    lower_bound = -1.96/np.sqrt(len(vector))

    # Cambio el color del punto si supera el limite
    col = np.where(
        autocorrelaciones[atype] < lower_bound,'b',
        np.where(autocorrelaciones[atype] > upper_bound, 'b', 'green'))

    # Creo los bastones
    plt.vlines(x=autocorrelaciones['lag'], ymin=0, ymax= autocorrelaciones[atype], color='black', alpha=0.4)

    # Creo los puntos y les asigno color segun pasen o no la banda
    plt.scatter(x = autocorrelaciones['lag'], y = autocorrelaciones[atype], color=col, alpha=1, marker = '.')

    # Grafico los limites de la banda
    plt.axhline(y = 0, color = 'black')
    plt.axhline(y = upper_bound, color = 'green', ls = '--')
    plt.axhline(y = lower_bound, color = 'green', ls = '--')
    plt.axhspan(upper_bound, lower_bound, color='green', alpha=0.15)
    plt.axhspan(1, 2, color='grey', alpha=0.5)
    plt.axhspan(-1, -2, color='grey', alpha=0.5)
    plt.ylim(-1.15,1.15)




# ------------------------------------------------------------------------------------
# Funcion resid_chek()
# Grafica la comprobacion de supuestos para los modelos arima

def resid_check(residuos_sin_estandarizar, ds):

    residuos = (residuos_sin_estandarizar - np.mean(residuos_sin_estandarizar))/ np.std(residuos_sin_estandarizar)
        
    # Histograma
    plt.subplot(2,2,1)
    sns.histplot(residuos, bins=20)

    # Test de normalidad
    ks = stats.kstest(residuos,'norm').pvalue
    if ks < 0.0001:
        ks = '< 0.0001'
    else:
        ks = round(ks, 4)

    ax = plt.gca()

    plt.text(0.98, 0.9, 
         f'P-value test de\nnormalidad K-S: {ks}', fontsize=6, color='black',
         transform=ax.transAxes,
         ha = 'right', va = 'top')

    # Serie de los residuos
    plt.subplot(2,2,2)
    sns.lineplot(x = ds, y = residuos)
    plt.axhline(y = 3, color = 'black', ls = '--')
    plt.axhline(y = -3, color = 'black', ls = '--')
    plt.ylabel('')
    outlayers = ((residuos > 3) | (residuos < -3))
    plt.scatter(
         x = ds[outlayers],
         y = residuos[outlayers],
         color = 'red', zorder = 10, marker='.')

    # Autocorrelaciones
    plt.subplot(2,2,3)
    autocorr_plot(residuos, lags=30)
    plt.title('')

    # Test de Ljung-box
    p_value = sm.stats.acorr_ljungbox(residuos, lags= range(1,20) , return_df=True)['lb_pvalue'].min()

    plt.text(10, -0.85, 
            f'Test de Ljung-Box\nMenor p-value: {round(p_value,4)}', fontsize=8, color='black')


    plt.subplot(2,2,4)
    autocorr_plot(residuos, lags=30, atype='pacf')
    plt.title('')

    plt.show()