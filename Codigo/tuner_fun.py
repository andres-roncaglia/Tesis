# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para cargar las claves
import creds

# Para XGBoost
from xgboost import XGBRegressor

# Para LightGBM
import lightgbm as lgb

# Para LSTM
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss

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

# Para calcular el interval score
from Funciones import interval_score


# ---------------------------------------- FUNCIONES ------------------------------------



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

def fit_pred_tgpt(df, h, time_col, target_col, freq, alpha, kwargs, devolver_tiempo = False):

    # Debemos garantizar que los int se mantengan como tal
    kwargs = {k: int(v) if isinstance(v, str) and v.isdigit() else v
    for k, v in kwargs.items()}

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        
    # Ajustamos el modelo y pronosticamos 
    forecaster = nixtla_client.forecast(df = df, h = h, time_col= time_col,
            target_col= target_col, freq= freq, level=[(1-alpha/2)*100], **kwargs)
        
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
    parametros['loss'] = MQLoss(level = [(1-alpha/2)*100])

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

    # Dividimos en entrenamiento, validacion y testeo
    corte = len(datos_xgb)-long_pred

    datos_xgb_fulltrain = datos_xgb[:corte]
    datos_xgb_test = datos_xgb[corte:]
    datos_xgb_train, datos_xgb_val = temporal_train_test_split(datos_xgb_fulltrain, test_size=0.2)

    # Extraemos las características
    X_fulltrain = datos_xgb_fulltrain[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]
    X_test = datos_xgb_test[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]
    X_train = datos_xgb_train[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]
    X_val = datos_xgb_val[['month', 'year', 'promedio_3_meses', 'desvio_3_meses', 'lag_1', 'lag_2', 'lag_12']]

    # Calculamos los cuantiles a pronosticar
    q_lower = alpha/2
    q_upper = 1-alpha/2

    # Por un error de train, debemos garantizar que los int y los floats se mantengan como tal
    kwargs = {k: int(v) if isinstance(v, float) and v.is_integer() else v
    for k, v in kwargs.items()}

    # Definimos los modelos
    model = XGBRegressor(**kwargs)

    # Ajustamos los modelos y pronosticamos

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar

    # Primero debemos obtener la distribucion de los residuos para el calculo de los intervalos
    model.fit(X_train, datos_xgb_train['y'])
    pred_val = model.predict(X_val)
    residuals = np.abs(datos_xgb_val['y'] - pred_val)

    # Calculamos el cuantil superior
    q = np.quantile(residuals, q_upper)

    # Volvemos a ajustar el modelo, ahora con todos lo datos de entrenamiento
    model.fit(X_fulltrain, datos_xgb_fulltrain['y'])
    pred = model.predict(X_test)

    # Calculamos los intervalos usando corformal predictions
    # Cuantil inferior
    pred_low = pred - q
    # Cuantil superior
    pred_upper = pred + q
        
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
    train_y, test_y = temporal_train_test_split(datos_train, test_size=0.15)
    
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
            forecast = fit_pred_tgpt(df = train_y, h = len(test_y), time_col= 'ds', target_col= 'y', freq= 'M', alpha = alpha, kwargs=kwargs)
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
        forecast, tiempo = fit_pred_tgpt(df = datos_train, h = len(datos_test), time_col= 'ds', target_col= 'y', freq= 'M', alpha=alpha, kwargs=kwargs, devolver_tiempo=True)
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

    # Devolvemos las predicciones
    return forecast, mape_final, score_final, tiempo, grilla

