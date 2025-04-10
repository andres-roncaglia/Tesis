# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para realizar consultas a la base de datos
import urllib.parse
import requests

# Para guardar y cargar los modelos
from joblib import dump, load

# Para cargar las claves
import creds

# Para XGBoost
from xgboost import XGBRegressor, DMatrix, train

# Para LightGBM
import lightgbm as lgb

# Para LSTM
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MQLoss

# Para time GPT
from nixtla import NixtlaClient
nixtla_client = NixtlaClient(api_key= creds.api_key)
import logging
logging.basicConfig(level=logging.WARNING) # Elimina los mensajes de INFO cuando se ajustan modelos


# Para calcular el MAPE
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# Para dividir el test de entrenamiento y testeo
from sktime.split import temporal_train_test_split

# Para medir el tiempo que tarda en ajustar los modelos
import time

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

def save_env(filename="environment.pkl"):
    global_vars = {k: v for k, v in globals().items() if not k.startswith("__") and is_pickleable(v)}
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
        global_vars = pickle.load(f)
        globals().update(global_vars)


# ------------------------------------------------------------------------------------

# Funcion dict_expand()
# Expande un diccionario transformandolo en un dataframe con todas las combinaciones de parametros
# argumentos:
# - parametros : Diccionario
# salida: Pandas Dataframe

def dict_expand(parametros):
    nombre_cols = list(parametros.keys())

    # Expandimos la grilla de parametros para evaluar todas las opciones
    ## Paso 1: Pasar los valores del diccionario como vectores de una lista
    grilla_lista = list(parametros.values())

    ## Paso 2: Expandimos la lista por todos los parametros. '*' sirve para desempaquetar los elementos de la lista, en lugar de pasarse como '[elem1, elem2]' se pasan como 'elem1, elem2'
    grilla_expan = list(map(np.ravel, np.meshgrid(*grilla_lista)))

    ## Paso 3: Guardamos todo como un Dataframe
    grilla = pd.DataFrame(np.array(grilla_expan).T, columns= nombre_cols)

    return grilla


# ------------------------------------------------------------------------------------

# Funcion fit_pred_tgpt()
# Ajusta un modelo de TimeGPT y pronostica
# argumentos:
# - Los mismos que nixtla_client.forecaster
# - devolver_tiempo : Delvolver el tiempo que tardo en ajustar o no
# salida: Pandas Dataframe con estimacion puntual y probabilistica, y opcionalmente el tiempo como variable numerica

def fit_pred_tgpt(df, h, time_col, target_col, freq, level, kwargs, devolver_tiempo = False):

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        
    # Ajustamos el modelo y pronosticamos 
    forecaster = nixtla_client.forecast(df = df, h = h, time_col= time_col,
            target_col= target_col, freq= freq, level=level, **kwargs)
        
    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Renombramos las columnas para generalizar
    forecaster.columns = ['ds', 'pred', 'upper', 'lower']

    if devolver_tiempo:
        return forecaster, tiempo
    else :
        return forecaster

# ------------------------------------------------------------------------------------

# Funcion fit_pred_lstm()
# Ajusta un modelo de LSTM y pronostica
# argumentos:
# - datos : datos de entrenamiento
# - long_pred : Horizonte de pronostico
# - kwargs : Parametros de un modelo LSTM 
# - alpha : Nivel de significacion de los intervalos 
# - devolver_tiempo : Delvolver el tiempo que tardo en ajustar o no
# salida: Pandas Dataframe con estimacion puntual y probabilistica, y opcionalmente el tiempo como variable numerica

def fit_pred_lstm(datos, long_pred, kwargs, alpha, devolver_tiempo = False):

    # Agregamos la columna unique_id necesaria por Neuralforecast
    datos_lstm = datos.copy()
    datos_lstm['unique_id'] = 0

    # Agregamos a los argumentos algunos otros necesarios
    parametros = kwargs.copy()
    parametros['h'] = long_pred
    parametros['input_size'] = long_pred*3
    parametros['loss'] = MQLoss(level = [1-alpha/2])

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        
    # Ajustamos el modelo y pronosticamos 
    model = NeuralForecast(models=[LSTM(**parametros)], freq='M')
    model.fit(df = datos_lstm)
    forecaster = model.predict()

    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Renombramos las columnas para generalizar
    forecaster.drop('unique_id', axis=1, inplace=True)
    forecaster.columns = ['ds', 'pred', 'upper', 'lower']

    if devolver_tiempo:
        return forecaster, tiempo
    else :
        return forecaster

# ------------------------------------------------------------------------------------

# Funcion quantile_loss()
# Función auxiliar para que XGB pronostique los quantiles
# argumentos:
# - q : Cuantil a pronosticar
# salida: Objetivo de prediccion

def quantile_loss(q):
    def objective(y_pred, d_train):
        y_true = d_train.get_label()
        err = y_true - y_pred
        grad = np.where(err < 0, -q, 1 - q)
        hess = np.ones_like(y_true)  
        return grad, hess
    return objective

# ------------------------------------------------------------------------------------

# Funcion fit_pred_xgb()
# Ajusta un modelo de XGBoost y pronostica
# argumentos:
# - datos : Conjunto de datos entero con columnas 'ds' e 'y'
# - long_pred : Horizonte de pronostico
# - kwargs : Diccionario con los argumentos a tunear de XGBoost
# - devolver_tiempo : Delvolver el tiempo que tardo en ajustar o no
# salida: Pandas Dataframe con estimacion puntual y probabilistica, y opcionalmente el tiempo como variable numerica

def fit_pred_xgb(datos, long_pred, alpha, kwargs, devolver_tiempo = False):

    # Creamos una copia del dataset
    datos_xgb = datos.copy()

    # Calculamos las características a usar
    datos_xgb['month'] = datos_xgb['ds'].dt.month
    datos_xgb['year'] = datos_xgb['ds'].dt.year
    datos_xgb["promedio_3_meses"] = datos_xgb["y"].shift(1).rolling(window=3).mean() # Promedio en una ventana de 3 meses
    datos_xgb["desvio_3_meses"] = datos_xgb["y"].shift(1).rolling(window=3).std() # Desvio en una ventana de 3 meses
    datos_xgb["lag_1"] = datos_xgb["y"].shift(1)
    datos_xgb["lag_2"] = datos_xgb["y"].shift(2)
    datos_xgb["lag_12"] = datos_xgb["y"].shift(12)

    # Dividimos en entrenamiento y testeo
    corte = len(datos_xgb)-long_pred

    datos_xgb_train = datos_xgb[:corte]
    datos_xgb_test = datos_xgb[corte:]

    # Extraemos las características
    X_train = datos_xgb_train[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]
    X_test = datos_xgb_test[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]

    # Calculamos los cuantiles a pronosticar
    q_lower = alpha/2
    q_upper = 1-alpha/2

    # Transformamos los datos de entrenamiento en un formato tal que xgb.train los pueda usar
    dtrain = DMatrix(X_train, label=datos_xgb_train['y'])

    # Añadimos unos argumentos extras a kwargs para que xgb.train funcione correctamente
    train_args = kwargs.copy()
    train_args.pop('n_estimators')
    train_args['objective'] = "reg:squarederror"
    train_args['base_score'] = np.median(datos_xgb_train['y'])

    # Por un error de train, debemos garantizar que los int y los floats se mantengan como tal
    train_args = {k: int(v) if isinstance(v, float) and v.is_integer() else v
    for k, v in train_args.items()}
    kwargs = {k: int(v) if isinstance(v, float) and v.is_integer() else v
    for k, v in kwargs.items()}

    # Definimos los modelos
    model = XGBRegressor(**kwargs)

    # Ajustamos los modelos y pronosticamos

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
    
    # Media
    model.fit(X_train, datos_xgb_train['y'])
    pred = model.predict(X_test)

    # Cuantil inferior
    model_low = train(dtrain= dtrain, obj=quantile_loss(q_lower), num_boost_round= kwargs['n_estimators'], params= train_args)
    pred_low = model_low.predict(DMatrix(X_test))
    
    # Cuantil superior
    model_upper = train(dtrain= dtrain, obj=quantile_loss(q_upper), num_boost_round= kwargs['n_estimators'], params= train_args)
    pred_upper = model_upper.predict(DMatrix(X_test))

        
    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Creamos un dataframe
    forecaster = pd.DataFrame({
        'ds' : datos_xgb_test['ds'],
        'pred' : pred,
        'lower' : pred_low,
        'upper' : pred_upper
    })

    if devolver_tiempo:
        return forecaster, tiempo
    else :
        return forecaster


# ------------------------------------------------------------------------------------

# Funcion fit_pred_lightgbm()
# Ajusta un modelo de LightGBM y pronostica
# argumentos:
# - datos : Conjunto de datos entero con columnas 'ds' e 'y'
# - long_pred : Horizonte de pronostico
# - kwargs : Diccionario con los argumentos a tunear de XGBoost
# - devolver_tiempo : Delvolver el tiempo que tardo en ajustar o no
# salida: Pandas Dataframe con estimacion puntual y probabilistica, y opcionalmente el tiempo como variable numerica

def fit_pred_lightgbm(datos, long_pred, alpha, kwargs, devolver_tiempo = False):

    # Creamos una copia del dataset
    datos_xgb = datos.copy()

    # Calculamos las características a usar
    datos_xgb['month'] = datos_xgb['ds'].dt.month
    datos_xgb['year'] = datos_xgb['ds'].dt.year
    datos_xgb["promedio_3_meses"] = datos_xgb["y"].shift(1).rolling(window=3).mean() # Promedio en una ventana de 3 meses
    datos_xgb["desvio_3_meses"] = datos_xgb["y"].shift(1).rolling(window=3).std() # Desvio en una ventana de 3 meses
    datos_xgb["lag_1"] = datos_xgb["y"].shift(1)
    datos_xgb["lag_2"] = datos_xgb["y"].shift(2)
    datos_xgb["lag_12"] = datos_xgb["y"].shift(12)

    # Dividimos en entrenamiento y testeo
    corte = len(datos_xgb)-long_pred

    datos_xgb_train = datos_xgb[:corte]
    datos_xgb_test = datos_xgb[corte:]

    # Extraemos las características
    X_train = datos_xgb_train[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]
    X_test = datos_xgb_test[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]

    # Calculamos los cuantiles a pronosticar
    q_lower = alpha/2
    q_upper = 1-alpha/2

    # Debemos garantizar que los int y los floats se mantengan como tal
    kwargs = {k: int(v) if isinstance(v, float) and v.is_integer() else v
    for k, v in kwargs.items()}

    # Definimos los modelos
    model = lgb.LGBMRegressor(**kwargs)
    model_low = lgb.LGBMRegressor(objective = 'quantile', alpha = q_lower,**kwargs)
    model_upper = lgb.LGBMRegressor(objective = 'quantile', alpha = q_upper,**kwargs)

    # Ajustamos los modelos y pronosticamos

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
    
    # Media
    model.fit(X_train, datos_xgb_train['y'])
    pred = model.predict(X_test)

    # Cuantil inferior
    model_low.fit(X_train, datos_xgb_train['y'])
    pred_low = model_low.predict(X_test)
    
    # Cuantil superior
    model_upper.fit(X_train, datos_xgb_train['y'])
    pred_upper = model_upper.predict(X_test)

        
    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Creamos un dataframe
    forecaster = pd.DataFrame({
        'ds' : datos_xgb_test['ds'],
        'pred' : pred,
        'lower' : pred_low,
        'upper' : pred_upper
    })

    if devolver_tiempo:
        return forecaster, tiempo
    else :
        return forecaster


# ------------------------------------------------------------------------------------

# Funcion Tuner()
# Busca el mejor modelo dentro los parametros especificados
# argumentos:
# - forecaster_fun : Funcion que se usa para ajustar el modelo
# - datos : pandas Dataframe con 2 columnas, fecha y respuesta
# - parametros : diccionario con opciones del modelo
# - metrica : Metrica para elegir el mejor modelo, MAPE o Interval Score
# - alpha : Nivel de significacion para los intervalos
# - long_pred : Horizonte de pronostico
# salida: 
# - forecaster : Pronosticos e intervalos
# - mape_final : MAPE del modelo final
# - score_final : Interval Score del modelo final
# - tiempo : Tiempo que tardó el ultimo modelo en ajustarse
# - grilla : Grilla con las metricas de los modelos probados sobre el conjunto de validacion

def Tuner(forecaster_fun, datos, parametros = {}, metrica = 'MAPE', alpha = 0.05, long_pred = 12):

    # Dividimos el conjunto de datos que queremos pronosticar
    corte = len(datos)-long_pred

    datos.columns = ['ds', 'y']

    datos_train = datos[:corte]
    datos_test = datos[corte:]

    # Dado que estamos ajustando parametros, no podemos usar el conjunto de entrenamiento en su totalidad, debemos particionarlo para evitar el sobreajuste
    train_y, test_y = temporal_train_test_split(datos_train, test_size=0.1)
    
    # Creamos vectores para guardar los resultados
    mapes = []
    scores = []

    # Expandimos la grilla de parametros
    grilla = dict_expand(parametros)

    # Vamos a probar cada combinacion de parametros 
    for j in range(0,grilla.shape[0]):

        # Primero pasamos la fila como diccionario para usar los argumentos
        kwargs = grilla.iloc[j].to_dict()
        
        # Ajustamos el modelo
        if forecaster_fun == 'XGBoost':
            forecast = fit_pred_xgb(datos = datos_train, long_pred = len(test_y), alpha = alpha, kwargs = kwargs)
        elif forecaster_fun == 'TimeGPT':
            forecast = fit_pred_tgpt(df = train_y, h = len(test_y), time_col= 'ds', target_col= 'y', freq= 'M', level=[1-alpha/2], kwargs=kwargs)
        elif forecaster_fun == 'LSTM':
            forecast = fit_pred_lstm(datos= train_y, long_pred= len(test_y), kwargs= kwargs, alpha=alpha)
        elif forecaster_fun == 'LightGBM':
            forecast = fit_pred_lightgbm(datos = datos_train, long_pred = len(test_y), alpha = alpha, kwargs = kwargs)


        # Calculamos MAPE
        mape = mean_absolute_percentage_error(test_y['y'], forecast['pred'])
        mapes.append(mape)

        # Calculamos interval score
        score = interval_score(obs=test_y['y'], lower= forecast['lower'], upper= forecast['upper'], alpha = alpha)
        scores.append(score)


    # Una vez probamos todas las opciones, vemos con cual modelo se obtuvo el menor error
    if metrica == 'MAPE': 
        mejor_combinacion = mapes.index(np.nanmin(mapes))
    else :
        mejor_combinacion = scores.index(np.nanmin(scores))
        

    # Por ultimo ajustamos el mejor modelo con todo el conjunto de entrenamiento:

    kwargs = grilla.iloc[mejor_combinacion].to_dict()


    # Ajustamos el modelo
    if forecaster_fun == 'XGBoost':
        forecast, tiempo = fit_pred_xgb(datos = datos, long_pred= len(datos_test), alpha = alpha, kwargs = kwargs, devolver_tiempo=True)
    elif forecaster_fun == 'TimeGPT':
        forecast, tiempo = fit_pred_tgpt(df = datos_train, h = len(datos_test), time_col= 'ds', target_col= 'y', freq= 'M', level=[1-alpha/2], kwargs=kwargs, devolver_tiempo=True)
    elif forecaster_fun == 'LSTM':
        forecast, tiempo = fit_pred_lstm(datos= datos_train, long_pred= len(datos_test), kwargs= kwargs, alpha=alpha, devolver_tiempo=True)
    elif forecaster_fun == 'LightGBM':
        forecast, tiempo = fit_pred_lightgbm(datos = datos, long_pred= len(datos_test), alpha = alpha, kwargs = kwargs, devolver_tiempo=True)
    
    
    # Agregamos a la grilla los mapes de cada combinacion
    grilla['MAPE'] = mapes
    mape_final = mean_absolute_percentage_error(datos_test['y'], forecast['pred'])

    # Agregamos a la grilla los scores de cada combinacion
    grilla['Interval Scores'] = scores
    score_final = interval_score(obs=datos_test['y'], lower=forecast['lower'], upper=forecast['upper'], alpha = alpha)

    if metrica == 'MAPE':
        grilla['Seleccionado'] = mapes == np.nanmin(mapes)
    else:
        grilla['Seleccionado'] = scores == np.nanmin(scores)

    # Graficamos el pronostico
    plt.plot(datos['ds'], datos['y'])
    sns.lineplot(x = forecast['ds'], y= forecast['pred'], color = 'red', label = 'Prediccion')
    plt.fill_between(forecast['ds'], forecast['lower'], forecast['upper'], color = 'red', alpha = 0.3)
        
    # Devolvemos las predicciones
    return forecast, mape_final, score_final, tiempo, grilla



# ------------------------------------------------------------------------------------
# PROBABLEMENTE NO LA USE MAS, BORRAR EN EL FUTURO
# Funcion tuner()


def tuner(forecaster_fun, datos, parametros = '', metrica = 'MAPE', alpha = 0.05, long_pred = 12):

    # Dividimos el conjunto de datos que queremos pronosticar
    corte = len(datos)-long_pred

    datos.columns = ['ds', 'y']

    datos_train = datos[:corte]
    datos_test = datos[corte:]

    # Dado que estamos ajustando parametros, no podemos usar el conjunto de entrenamiento en su totalidad, debemos particionarlo para evitar el sobreajuste
    train_y, test_y = temporal_train_test_split(datos_train['y'], test_size=0.1)

    num_pred = ForecastingHorizon(test_y.index, is_relative=False)

    # Si no se definen parametros, simplemente se devuelve el modelo base ajustado
    if not isinstance(parametros, dict):
        
        fh = ForecastingHorizon(datos_test.index, is_relative=False)

        timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        forecaster = forecaster_fun().fit(datos_train['y'])
          
        # Obtenemos predicciones
        pred = forecaster.predict(fh)
        pred_int = forecaster.predict_interval(fh, coverage=1-alpha*2)
        timer_final = time.time()
        tiempo = timer_final - timer_comienzo

        # Calculamos MAPE
        mape_final = mean_absolute_percentage_error(datos_test['y'], pred)

        # Calculamos Interval Score
        pred_int.columns = ['lower', 'upper']
        score_final = interval_score(obs=datos_test['y'], lower=pred_int['lower'], upper=pred_int['upper'], alpha = alpha)

        # Graficamos el pronostico
        datos = pd.concat([datos_train, datos_test])
        plt.plot(datos['ds'], datos['y'])
        sns.lineplot(x = datos['ds'], y= pred, color = 'red', label = 'Prediccion')
        plt.fill_between(datos_test['ds'], pred_int['lower'], pred_int['upper'], color = 'red', alpha = 0.3, label = f'IC: {1-alpha*2}%')

        # Devolvemos las predicciones
        return pred, mape_final, score_final, tiempo, []


    
    mapes = []
    scores = []
    nombre_cols = list(parametros.keys())

    # Expandimos la grilla de parametros para evaluar todas las opciones
    ## Paso 1: Pasar los valores del diccionario como vectores de una lista
    grilla_lista = list(parametros.values())

    ## Paso 2: Expandimos la lista por todos los parametros. '*' sirve para desempaquetar los elementos de la lista, en lugar de pasarse como '[elem1, elem2]' se pasan como 'elem1, elem2'
    grilla_expan = list(map(np.ravel, np.meshgrid(*grilla_lista)))

    ## Paso 3: Guardamos todo como un Dataframe
    grilla = pd.DataFrame(np.array(grilla_expan).T, columns= nombre_cols)

    # Vamos a probar cada combinacion de filas 
    for j in range(0,grilla.shape[0]):

        # Primero pasamos la fila como diccionario para usar los argumentos
        kwargs = grilla.iloc[j].to_dict()
            

        # TimeGPT no necesita una funcion para ajustar el modelo
        
        if forecaster_fun.__name__ == 'forecast':
            gpttrain_y = pd.DataFrame({'y': train_y, 'ds': datos_train['ds'][:len(train_y)]})
            forecaster = forecaster_fun(df = gpttrain_y, h = len(test_y), time_col= 'ds',
            target_col= 'y', freq= 'M', level=[1-alpha*2], **kwargs)

            # Calculamos MAPE
            mape = mean_absolute_percentage_error(test_y, forecaster['TimeGPT'])
            mapes.append(mape)

            # Calculamos interval score
            score = interval_score(obs=test_y, lower= forecaster[f'TimeGPT-lo-{1-alpha*2}'], upper= forecaster[f'TimeGPT-hi-{1-alpha*2}'], alpha = alpha)
            scores.append(score)

            continue


        # Luego especificamos el pronosticador
        forecaster = forecaster_fun(**kwargs)


        # Ajustamos el modelo
        try:
            forecaster.fit(train_y)
          
            # Obtenemos predicciones
            pred = forecaster.predict(num_pred)
            pred_int = forecaster.predict_interval(num_pred, coverage=1-alpha*2)

            # Calculamos MAPE
            mape = mean_absolute_percentage_error(test_y, pred)
            mapes.append(mape)

            # Calculamos Interval Score
            pred_int.columns = ['lower', 'upper']
            score = interval_score(obs=test_y, lower=pred_int['lower'], upper=pred_int['upper'], alpha = 1-alpha*2)
            scores.append(score)
        except:
            # Si el modelo falla en ajustar, asignamos NaN
            mapes.append(np.NaN)
            scores.append(np.NaN)

    # Una vez probamos todas las opciones, vemos con cual modelo se obtuvo el menor error
    if metrica == 'MAPE': 
        mejor_combinacion = mapes.index(np.nanmin(mapes))
    else :
        mejor_combinacion = scores.index(np.nanmin(scores))
        

    # Por ultimo ajustamos el mejor modelo con todo el conjunto de entrenamiento:

    kwargs = grilla.iloc[mejor_combinacion].to_dict()

    # Time GPT no necesita llamar a una funcion para ajustar el modelo

    if forecaster_fun.__name__ == 'forecast':
        timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        
        forecaster = forecaster_fun(df = datos_train, h = len(datos_test), time_col= 'ds',
            target_col= 'y', freq= 'M', level=[1-alpha*2], **kwargs)
        
        timer_final = time.time()
        tiempo = timer_final - timer_comienzo

        pred = forecaster['TimeGPT']
        pred_int = forecaster[[f'TimeGPT-lo-{1-alpha*2}', f'TimeGPT-hi-{1-alpha*2}']]
    
    else: 
        forecaster = forecaster_fun(**kwargs)
        
        timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        forecaster.fit(datos_train['y'])
            
        # Obtenemos predicciones
        fh = ForecastingHorizon(datos_test.index, is_relative=False)
        pred = forecaster.predict(fh)
        pred_int = forecaster.predict_interval(num_pred, coverage=1-alpha*2)

        timer_final = time.time()
        tiempo = timer_final - timer_comienzo
    
    # Agregamos a la grilla los mapes de cada combinacion
    grilla['MAPE'] = mapes
    mape_final = mean_absolute_percentage_error(datos_test['y'], pred)

    # Agregamos a la grilla los scores de cada combinacion
    grilla['Interval Scores'] = scores
    pred_int.columns = ['lower', 'upper']
    score_final = interval_score(obs=datos_test['y'], lower=pred_int['lower'], upper=pred_int['upper'], alpha = alpha)

    if metrica == 'MAPE':
        grilla['Seleccionado'] = mapes == np.nanmin(mapes)
    else:
        grilla['Seleccionado'] = scores == np.nanmin(scores)

    # Graficamos el pronostico
    datos = pd.concat([datos_train, datos_test])
    plt.plot(datos['ds'], datos['y'])
    if forecaster_fun.__name__ == 'forecast':
        sns.lineplot(x = forecaster['ds'], y= pred, color = 'red', label = 'Prediccion')
        plt.fill_between(forecaster['ds'], pred_int['lower'], pred_int['upper'], color = 'red', alpha = 0.3)

        print(pred_int)   
    else : 
        sns.lineplot(x = datos_test['ds'], y= pred, color = 'red', label = 'Prediccion')
        plt.fill_between(datos_test['ds'], pred_int['lower'], pred_int['upper'], color = 'red', alpha = 0.3)

    # Devolvemos las predicciones
    return pred, mape_final, score_final, tiempo, grilla