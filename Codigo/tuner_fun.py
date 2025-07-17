# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para cargar las claves
from Codigo import creds

# Para ARIMA
from pmdarima import auto_arima

# Para XGBoost
from xgboost import XGBRegressor

# Para LightGBM
import lightgbm as lgb

# Para conformal predictions
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

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

# Para medir el tiempo que tarda en ajustar los modelos
import time

# Para calcular el interval score
from Codigo.Funciones import interval_score


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

# Funcion fit_pred_arima()
# Ajusta un modelo de ARIMA y pronostica
# argumentos:
# - datos : Conjunto de datos entero con columnas 'ds' e 'y'
# - long_pred : Horizonte de pronostico
# - kwargs : Diccionario con los argumentos a tunear de XGBoost
# - devolver_tiempo : Delvolver el tiempo que tardo en ajustar o no
# salida: Pandas Dataframe con estimacion puntual y probabilistica, y opcionalmente el tiempo como variable numerica

def fit_pred_arima(datos, long_pred, alpha, exog, kwargs):

    datos_arima = datos.copy()

    # Si tenemos variables exogenas las agregamos
    if exog.shape[0] != 0:
        kwargs['X'] = exog.head(len(exog)-long_pred)
        exog_test = exog.tail(long_pred)
    else :
        exog_test = None

    # Dividimos en entrenamiento, validacion y testeo
    datos_arima_train = datos_arima.head(len(datos_arima)-long_pred)
    datos_arima_test = datos_arima.tail(long_pred)

    # Ajustamos el modelo
    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar

    modelo = auto_arima(datos_arima_train['y'],
                        m = long_pred,
                        trace = False, 
                        error_action ='ignore',   
                        suppress_warnings = True, 
                        stepwise = True,
                        **kwargs)

    # Obtenemos predicciones
    pred, pred_int = modelo.predict(n_periods = long_pred, alpha = alpha, return_conf_int=True, X = exog_test)
    
    timer_final = time.time()
    tiempo = timer_final - timer_comienzo

    # Guardamos los pronosticos en un dataframe
    pred_int = pd.DataFrame(pred_int, columns=['lower', 'upper'])
    pred_int.reset_index(drop=True, inplace=True)

    pred_arima = pd.concat([
        datos_arima_test['ds'].reset_index(drop=True),
        pred.reset_index(drop=True),
        pred_int['lower'].reset_index(drop=True),
        pred_int['upper'].reset_index(drop=True)
    ], axis=1)

    pred_arima.columns = ['ds', 'pred', 'lower', 'upper']
    # Calculamos MAPE
    mape = mean_absolute_percentage_error(datos_arima_test['y'], pred_arima['pred'])

    # Calculamos Interval Score
    score = interval_score(obs=datos_arima_test['y'], lower=pred_arima['lower'], upper=pred_arima['upper'], alpha = alpha)

    return pred_arima, tiempo, modelo, mape, score


# ------------------------------------------------------------------------------------

# Funcion fit_pred_tgpt()
# Ajusta un modelo de TimeGPT y pronostica
# argumentos:
# - Los mismos que nixtla_client.forecaster
# - devolver_tiempo : Delvolver el tiempo que tardo en ajustar o no
# salida: Pandas Dataframe con estimacion puntual y probabilistica, y opcionalmente el tiempo como variable numerica

def fit_pred_tgpt(df, h, time_col, target_col, freq, alpha, kwargs, exog, devolver_tiempo = False, devolver_modelo = False):

    # Debemos garantizar que los int se mantengan como tal
    kwargs = {k: int(v) if isinstance(v, str) and v.isdigit() else v
    for k, v in kwargs.items()}

    # Si tenemos variables exogenas las agregamos
    if exog.shape[0] != 0:
        exog_names = list(exog.columns) # Obtenemos los nombres
        df = pd.concat([df.reset_index(drop=True), exog.reset_index(drop=True)], axis=1) # Agregamos las variables al dataset
        kwargs['hist_exog_list'] = exog_names

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar
        
    # Ajustamos el modelo y pronosticamos 
    forecaster = nixtla_client.forecast(df = df, h = h, time_col= time_col,
            target_col= target_col, freq= freq, level=[(1-alpha)*100], **kwargs)
        
    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Renombramos las columnas para generalizar
    forecaster.columns = ['ds', 'pred', 'upper', 'lower']

    model= np.NaN

    if devolver_tiempo & devolver_modelo:
        return forecaster, tiempo, model
    elif devolver_modelo:
        return forecaster, model
    elif devolver_tiempo:
        return forecaster, tiempo
    else:
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

def fit_pred_lstm(datos, long_pred, kwargs, alpha, freq, exog, devolver_tiempo = False, devolver_modelo = False):

    # Agregamos la columna unique_id necesaria por Neuralforecast
    datos_lstm = datos.copy()
    datos_lstm['unique_id'] = 0

    # Debemos garantizar que los int y los floats se mantengan como tal
    kwargs = {
        k: int(v) if (
            (isinstance(v, float) and v.is_integer()) or 
            (isinstance(v, str) and v.isdigit())
        ) else v
        for k, v in kwargs.items()
    }

    # Agregamos a los argumentos algunos otros necesarios
    parametros = kwargs.copy()
    parametros['h'] = long_pred
    parametros['input_size'] = long_pred*3
    parametros['loss'] = MQLoss(level = [(1-alpha)*100])
    
    # Si tenemos variables exogenas las agregamos
    if exog.shape[0] != 0:
        exog_names = list(exog.columns) # Obtenemos los nombres
        datos_lstm = pd.concat([datos_lstm.reset_index(drop=True), exog.reset_index(drop=True)], axis=1) # Agregamos las variables al dataset
        parametros['hist_exog_list'] = exog_names

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar

    # Ajustamos el modelo y pronosticamos 
    model = NeuralForecast(models=[LSTM(**parametros)], freq=freq)
    model.fit(df = datos_lstm, val_size=long_pred)
    forecaster = model.predict()

    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Renombramos las columnas para generalizar
    forecaster.drop('unique_id', axis=1, inplace=True)
    
    forecaster.columns = ['ds', 'pred', 'upper', 'lower']

    if devolver_tiempo & devolver_modelo:
        return forecaster, tiempo, model
    elif devolver_modelo:
        return forecaster, model
    elif devolver_tiempo:
        return forecaster, tiempo
    else:
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

def fit_pred_xgb(datos, long_pred, alpha, kwargs, caracteristicas, exog, devolver_tiempo = False, devolver_modelo = False):

    # Creamos una copia del dataset
    datos_xgb = datos.copy()

    # Agregamos las caracteristicas
    datos_xgb = pd.concat([datos_xgb.reset_index(drop=True), caracteristicas.reset_index(drop=True)], axis=1)

    # Si tenemos variables exogenas las agregamos a las caracteristicas
    if exog.shape[0] != 0:
        datos_xgb = pd.concat([datos_xgb.reset_index(drop=True), exog.reset_index(drop=True)], axis=1)

    # Dividimos en entrenamiento, validacion y testeo
    datos_xgb_train = datos_xgb.head(len(datos_xgb)-long_pred)
    datos_xgb_test = datos_xgb.tail(long_pred)

    # Extraemos las características
    X_train = datos_xgb_train.drop(columns = ['ds','y'])
    X_test = datos_xgb_test.drop(columns = ['ds','y'])

    # Debemos garantizar que los int y los floats se mantengan como tal
    kwargs = {
        k: int(v) if (
            (isinstance(v, float) and v.is_integer()) or 
            (isinstance(v, str) and v.isdigit())
        ) else v
        for k, v in kwargs.items()
    }

    # Definimos los modelos
    model = XGBRegressor(**kwargs)
    
    # Definimos los bloques para validacion cruzada y el modelo para conformal predictions
    bt_blocks = BlockBootstrap(
        n_resamplings=30, n_blocks=int(min(np.floor(len(datos_xgb_train)/20), 10)), overlapping=False, random_state=kwargs['random_state'])

    mapie_enbpi = MapieTimeSeriesRegressor(
        model, method="enbpi", cv=bt_blocks, agg_function="mean", n_jobs=-1)

    # Ajustamos los modelos y pronosticamos

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar

    mapie_enbpi = mapie_enbpi.fit(X_train, datos_xgb_train['y'])

    pred, y_ic = mapie_enbpi.predict(X_test, alpha=alpha, ensemble=True)
        
    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    # Creamos un dataframe
    pred_lower = y_ic.squeeze()[:,0]
    pred_upper = y_ic.squeeze()[:,1]

    forecaster = pd.DataFrame({
        'ds' : datos_xgb_test['ds'],
        'pred' : pred,
        'lower' : pred_lower,
        'upper' : pred_upper
    })

    if devolver_tiempo & devolver_modelo:
        return forecaster, tiempo, model
    elif devolver_modelo:
        return forecaster, model
    elif devolver_tiempo:
        return forecaster, tiempo
    else:
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

def fit_pred_lightgbm(datos, long_pred, alpha, kwargs, caracteristicas, exog, devolver_tiempo = False, devolver_modelo = False):

    # Creamos una copia del dataset
    datos_xgb = datos.copy()

    # Agregamos las caracteristicas
    datos_xgb = pd.concat([datos_xgb.reset_index(drop=True), caracteristicas.reset_index(drop=True)], axis=1)

    # Si tenemos variables exogenas las agregamos a las caracteristicas
    if exog.shape[0] != 0:
        datos_xgb = pd.concat([datos_xgb.reset_index(drop=True), exog.reset_index(drop=True)], axis=1)

    # Dividimos en entrenamiento, validacion y testeo
    datos_xgb_train = datos_xgb.head(len(datos_xgb)-long_pred)
    datos_xgb_test = datos_xgb.tail(long_pred)

    # Extraemos las características
    X_train = datos_xgb_train.drop(columns = ['ds','y'])
    X_test = datos_xgb_test.drop(columns = ['ds','y'])

    # Debemos garantizar que los int y los floats se mantengan como tal
    kwargs = {k: int(v) if isinstance(v, float) and v.is_integer() else v
    for k, v in kwargs.items()}

    # Definimos los modelos
    model = lgb.LGBMRegressor(**kwargs)

    # Definimos los bloques para validacion cruzada y el modelo para conformal predictions
    bt_blocks = BlockBootstrap(
        n_resamplings=30, n_blocks=int(min(np.floor(len(datos_xgb_train)/20), 10)), overlapping=False, random_state=kwargs['random_state'])

    mapie_enbpi = MapieTimeSeriesRegressor(
        model, method="enbpi", cv=bt_blocks, agg_function="mean", n_jobs=-1)

    # Ajustamos los modelos y pronosticamos

    timer_comienzo = time.time() # Empiezo a medir cuanto tarda en ajustar

    mapie_enbpi = mapie_enbpi.fit(X_train, datos_xgb_train['y'])

    pred, y_ic = mapie_enbpi.predict(X_test, alpha=alpha, ensemble=True)
        
    timer_final = time.time()

    # Medimos el tiempo que llevo
    tiempo = timer_final - timer_comienzo

    pred_lower = y_ic.squeeze()[:,0]
    pred_upper = y_ic.squeeze()[:,1]

    # Creamos un dataframe
    forecaster = pd.DataFrame({
        'ds' : datos_xgb_test['ds'],
        'pred' : pred,
        'lower' : pred_lower,
        'upper' : pred_upper
    })

    if devolver_tiempo & devolver_modelo:
        return forecaster, tiempo, model
    elif devolver_modelo:
        return forecaster, model
    elif devolver_tiempo:
        return forecaster, tiempo
    else:
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

def Tuner(forecaster_fun, datos, parametros = {}, metrica = 'MAPE', alpha = 0.05, long_pred = 12, caracteristicas = pd.DataFrame(), exog = pd.DataFrame(), tgpt_freq = 'MS'):

    # Dividimos el conjunto de datos que queremos pronosticar
    datos_fulltrain = datos.head(len(datos)-long_pred)
    datos_test = datos.tail(long_pred)

    # Dado que estamos ajustando parametros, no podemos usar el conjunto de entrenamiento en su totalidad, debemos particionarlo para evitar el sobreajuste
    datos_train = datos_fulltrain.head(datos_fulltrain.shape[0] - long_pred)
    datos_val = datos_fulltrain.tail(long_pred)

    # Si tenemos variables exogenas tmb debemos dividir los conjuntos de entrenamiento, prueba y validación
    if exog.shape[0] != 0:
        exog_fulltrain = exog.head(len(exog)-long_pred)
        exog_test = exog.tail(long_pred)
        exog_train = exog_fulltrain.head(exog_fulltrain.shape[0] - long_pred)
        exog_val = exog_fulltrain.tail(long_pred)
    else :
        exog_fulltrain = pd.DataFrame()
        exog_test = pd.DataFrame()
        exog_train = pd.DataFrame()
        exog_val = pd.DataFrame()

    if forecaster_fun != 'ARIMA' and parametros != {}:

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
                forecast = fit_pred_xgb(datos = datos_fulltrain, long_pred = len(datos_val), alpha = alpha, kwargs = kwargs, caracteristicas=caracteristicas, exog= exog_fulltrain)
            elif forecaster_fun == 'TimeGPT':
                forecast = fit_pred_tgpt(df = datos_train, h = len(datos_val), time_col= 'ds', target_col= 'y', freq= tgpt_freq, alpha = alpha, kwargs=kwargs, exog= exog_train)
            elif forecaster_fun == 'LSTM':
                forecast = fit_pred_lstm(datos= datos_train, long_pred= len(datos_val), kwargs= kwargs, alpha=alpha, exog= exog_train, freq= tgpt_freq)
            elif forecaster_fun == 'LightGBM':
                forecast = fit_pred_lightgbm(datos = datos_fulltrain, long_pred = len(datos_val), alpha = alpha, kwargs = kwargs, caracteristicas=caracteristicas, exog= exog_fulltrain)


            # Calculamos MAPE
            mape = mean_absolute_percentage_error(datos_val['y'], forecast['pred'])
            mapes.append(mape)

            # Calculamos interval score
            score = interval_score(obs=datos_val['y'], lower= forecast['lower'], upper= forecast['upper'], alpha = alpha)
            scores.append(score)


        # Una vez probamos todas las opciones, vemos con cual modelo se obtuvo el menor error
        if metrica == 'MAPE': 
            mejor_combinacion = mapes.index(np.nanmin(mapes))
        else :
            mejor_combinacion = scores.index(np.nanmin(scores))
            

        # Por ultimo ajustamos el mejor modelo con todo el conjunto de entrenamiento:

        kwargs = grilla.iloc[mejor_combinacion].to_dict()


    # Ajustamos el modelo
    if forecaster_fun == 'ARIMA':
        forecast, tiempo, model, mape, score = fit_pred_arima(datos = datos, long_pred= long_pred, alpha = alpha, kwargs = parametros, exog= exog)
        return {'pred': forecast, 'mape': mape, 'score': score, 'tiempo': tiempo, 'modelo': model}
    elif forecaster_fun == 'XGBoost':
        forecast, tiempo, model = fit_pred_xgb(datos = datos, long_pred= long_pred, alpha = alpha, kwargs = kwargs, caracteristicas=caracteristicas, devolver_tiempo=True, devolver_modelo=True, exog= exog)
    elif forecaster_fun == 'TimeGPT':
        forecast, tiempo, model = fit_pred_tgpt(df = datos_fulltrain, h = long_pred, time_col= 'ds', target_col= 'y', freq= tgpt_freq, alpha=alpha, kwargs=parametros, devolver_tiempo=True, devolver_modelo=True, exog= exog_fulltrain)
        mape = mean_absolute_percentage_error(datos_test['y'], forecast['pred'])
        score = interval_score(obs=datos_test['y'], lower=forecast['lower'], upper=forecast['upper'], alpha = alpha)
        return {'pred': forecast, 'mape': mape, 'score': score, 'tiempo': tiempo}
    elif forecaster_fun == 'LSTM':
        forecast, tiempo, model = fit_pred_lstm(datos= datos_fulltrain, long_pred= long_pred, kwargs= kwargs, alpha=alpha, devolver_tiempo=True, devolver_modelo=True, exog= exog_fulltrain, freq= tgpt_freq)
    elif forecaster_fun == 'LightGBM':
        forecast, tiempo, model = fit_pred_lightgbm(datos = datos, long_pred= long_pred, alpha = alpha, kwargs = kwargs, caracteristicas=caracteristicas, devolver_tiempo=True, devolver_modelo=True, exog= exog)
    
    
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
    return {'pred': forecast, 'mape': mape_final, 'score': score_final, 'tiempo': tiempo, 'grilla': grilla, 'modelo': model}

