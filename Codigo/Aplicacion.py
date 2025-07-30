# ------------------------------- CARGA DE LIBRERIAS -------------------------------

# Para el manejo de estructuras de datos
import pandas as pd

# Para ajustar los modelos arima manualmente
from pmdarima.arima import ARIMA

# Cargamos funciones
from Codigo.Funciones import save_env, load_env
from Codigo.tuner_fun import Tuner

# Definimos una semilla
seed = 11072001

# Cargamos el ambiente
globals().update(load_env('Codigo/Ambiente/Amb_Aplicacion.pkl'))
globals().update(load_env('Codigo/Ambiente/modelos_aplicacion.pkl'))

# -------------------------------------------------------------------------
# ------------------------------- SERIE 1 ---------------------------------
# -------------------------------------------------------------------------


# ------------------------------- 1.1 CARGA DE DATOS -------------------------------

atenciones_guardia = pd.read_excel(io='Datos/Atenciones de guardia en el HNVV por patologías respiratorias (vigiladas por epidemiología).xlsx' )

# Aseguro que la columna fecha tenga el formato adecuado
atenciones_guardia['fec'] = pd.to_datetime(atenciones_guardia['fec'], format='%Y-%m-%d')

# Filtro las columnas importantes y las renombro
atenciones_guardia = atenciones_guardia[['fec', 'frec']]
atenciones_guardia.columns = ['ds','y']

# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2
long_pred = 12

# Creamos un dataframe donde vamos a guardar todos los resultados
metricas_1 = pd.DataFrame(columns=(['Modelo', 'Horizonte', 'MAPE', 'Interval Score', 'Tiempo']))

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
metricas_1.loc[len(metricas_1)] = ['ARIMA', '', resultados_1_arima['mape'], resultados_1_arima['score'], resultados_1_arima['tiempo']]

# Modelos manuales
atenciones_train = atenciones_guardia.head(len(atenciones_guardia)-long_pred).copy()
ds = atenciones_guardia.tail(long_pred)['ds'].reset_index(drop = True)

# Modelo 1
arima_atenciones_1 = ARIMA( 
    order=(0, 1, 1), 
    seasonal_order=(0,1,0,12))

arima_atenciones_1 = arima_atenciones_1.fit(atenciones_train['y'])

pred, pred_int = arima_atenciones_1.predict(n_periods = long_pred, alpha = alpha, return_conf_int=True)
pred_atenciones_1 = pd.DataFrame(pred_int, columns=['lower', 'upper'])
pred_atenciones_1['pred'] = pred.reset_index(drop = True)
pred_atenciones_1['ds'] = ds

# Modelo 2
arima_atenciones_2 = ARIMA( 
    order=(0, 1, 0), 
    seasonal_order=(0,1,1,12))

arima_atenciones_2 = arima_atenciones_2.fit(atenciones_train['y'])

pred, pred_int = arima_atenciones_2.predict(n_periods = long_pred, alpha = alpha, return_conf_int=True)
pred_atenciones_2 = pd.DataFrame(pred_int, columns=['lower', 'upper'])
pred_atenciones_2['pred'] = pred.reset_index(drop = True)
pred_atenciones_2['ds'] = ds


# ------------------------------- 1.3 XGBOOST -------------------------------


# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "eval_metric" : ['mape'],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.001, 0.1, 0.2],
    "n_estimators": [20, 50, 100, 150],
    "colsample_bytree": [0.7, 1.0],
}

# Calculamos las características a usar
caracteristicas_atenciones = pd.DataFrame({
    'month' : atenciones_guardia['ds'].dt.month,
    'year' : atenciones_guardia['ds'].dt.year,
    "promedio_3_meses" : atenciones_guardia["y"].shift(1).rolling(window=3).mean(),
    "desvio_3_meses" : atenciones_guardia["y"].shift(1).rolling(window=3).std(),
    "lag_1" : atenciones_guardia["y"].shift(1),
    "lag_2" : atenciones_guardia["y"].shift(2),
    "lag_12" : atenciones_guardia["y"].shift(12),
})


# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
resultados_1_xgb3 = Tuner(forecaster_fun= 'XGBoost', datos=atenciones_guardia.head(len(atenciones_guardia)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_atenciones.head(len(atenciones_guardia)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['XGBoost', long_pred, resultados_1_xgb3['mape'], resultados_1_xgb3['score'], resultados_1_xgb3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_1_xgb6 = Tuner(forecaster_fun= 'XGBoost', datos=atenciones_guardia.head(len(atenciones_guardia)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_atenciones.head(len(atenciones_guardia)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['XGBoost', long_pred, resultados_1_xgb6['mape'], resultados_1_xgb6['score'], resultados_1_xgb6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_1_xgb = Tuner(forecaster_fun= 'XGBoost', datos=atenciones_guardia, parametros=params, caracteristicas=caracteristicas_atenciones, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['XGBoost', long_pred, resultados_1_xgb['mape'], resultados_1_xgb['score'], resultados_1_xgb['tiempo']]


# ------------------------------- 1.4 LIGHTGBM -------------------------------

# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "verbose": [-1],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.001, 0.1, 0.2],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [2,4,8,16],
    "colsample_bytree": [0.7, 1.0],
}

# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
resultados_1_lgbm3 = Tuner(forecaster_fun= 'LightGBM', datos=atenciones_guardia.head(len(atenciones_guardia)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_atenciones.head(len(atenciones_guardia)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LightGBM', long_pred, resultados_1_lgbm3['mape'], resultados_1_lgbm3['score'], resultados_1_lgbm3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_1_lgbm6 = Tuner(forecaster_fun= 'LightGBM', datos=atenciones_guardia.head(len(atenciones_guardia)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_atenciones.head(len(atenciones_guardia)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LightGBM', long_pred, resultados_1_lgbm6['mape'], resultados_1_lgbm6['score'], resultados_1_lgbm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_1_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=atenciones_guardia, parametros=params, caracteristicas=caracteristicas_atenciones, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LightGBM', long_pred, resultados_1_lgbm['mape'], resultados_1_lgbm['score'], resultados_1_lgbm['tiempo']]

# ------------------------------- 1.5 LSTM -------------------------------

# Definimos los parametros a tunear
parametros = {
    'lags' : [1, 12, 24],
    'random_seed' : [seed],
    'epochs' : [300],
    'early_stop_patience' : [10],
    'units' : [[24,42],[32],[12,24]],
    'dropout' : [0.1, 0.3],
    'activation' : ['relu', 'tanh'],
    'learning_rate' : [0.001]
    }

# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
resultados_1_lstm3 = Tuner(forecaster_fun= 'LSTM', datos=atenciones_guardia.head(len(atenciones_guardia)-(12-long_pred)), parametros=parametros, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LSTM', long_pred, resultados_1_lstm3['mape'], resultados_1_lstm3['score'], resultados_1_lstm3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_1_lstm6 = Tuner(forecaster_fun= 'LSTM', datos=atenciones_guardia.head(len(atenciones_guardia)-(12-long_pred)), parametros=parametros, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LSTM', long_pred, resultados_1_lstm6['mape'], resultados_1_lstm6['score'], resultados_1_lstm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_1_lstm = Tuner(forecaster_fun= 'LSTM', datos=atenciones_guardia, parametros= parametros, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LSTM', long_pred, resultados_1_lstm['mape'], resultados_1_lstm['score'], resultados_1_lstm['tiempo']]

# ------------------------------- 1.6 TIMEGPT -------------------------------

# Tuneamos los parametros y ajustamos el modelo
resultados_1_gpt3 = Tuner(forecaster_fun= 'TimeGPT', datos=atenciones_guardia, alpha= alpha, long_pred = 3)
resultados_1_gpt6 = Tuner(forecaster_fun= 'TimeGPT', datos=atenciones_guardia, alpha= alpha, long_pred = 6)
resultados_1_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=atenciones_guardia, alpha= alpha, long_pred = 12)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['TimeGPT', 3, resultados_1_gpt3['mape'], resultados_1_gpt3['score'], resultados_1_gpt3['tiempo']]
metricas_1.loc[len(metricas_1)] = ['TimeGPT', 6, resultados_1_gpt6['mape'], resultados_1_gpt6['score'], resultados_1_gpt6['tiempo']]
metricas_1.loc[len(metricas_1)] = ['TimeGPT', 12, resultados_1_gpt['mape'], resultados_1_gpt['score'], resultados_1_gpt['tiempo']]


# -------------------------------------------------------------------------
# ------------------------------- SERIE 2 ---------------------------------
# -------------------------------------------------------------------------

# ------------------------------- 2.1 CARGA DE DATOS -------------------------------

# Cargamos los datos
trabajadores = pd.read_excel(io='Datos/trabajoregistrado_2502_estadisticas.xlsx', sheet_name= 'A.2.1', thousands='.', decimal=',', header=1, usecols='A,M', skipfooter=5, skiprows=84)

# Renombramos las columnas
trabajadores.columns = ['ds', 'y']

# Asignamos formato fecha
meses = {
    'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 
    'may': '05', 'jun': '06', 'jul': '07', 'ago': '08', 
    'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
}

trabajadores['ds'] = trabajadores['ds'].str.replace('*','')
trabajadores['ds'] = trabajadores['ds'].apply(
    lambda x: '01-' + x.replace(x.split('-')[0], meses.get(x.split('-')[0].lower(), '')).replace(x.split('-')[1], '20' + x.split('-')[1])
)

trabajadores['ds'] = pd.to_datetime(trabajadores['ds'], format='%d-%m-%Y')


# Eliminamos los datos del 2025 para tener solo años completos
trabajadores = trabajadores[trabajadores['ds'].dt.year != 2025]

# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2
long_pred = 12

# Creamos un dataframe donde vamos a guardar todos los resultados
metricas_2 = pd.DataFrame(columns=(['Modelo', 'Horizonte', 'MAPE', 'Interval Score', 'Tiempo']))


# ------------------------------- 2.2 ARIMA -------------------------------


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
resultados_2_arima = Tuner(forecaster_fun= 'ARIMA', datos=trabajadores, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['ARIMA', long_pred, resultados_2_arima['mape'], resultados_2_arima['score'], resultados_2_arima['tiempo']]

# Modelos manuales
trabajadores_trunc = trabajadores.head(len(trabajadores)-long_pred).copy()
ds = trabajadores.tail(long_pred)['ds'].reset_index(drop = True)

# Modelo 1
arima_trabajadores_1 = ARIMA( 
    order=(0,1,1), 
    seasonal_order=(1,1,0,12))

arima_trabajadores_1 = arima_trabajadores_1.fit(trabajadores_trunc['y'])

pred, pred_int = arima_trabajadores_1.predict(n_periods = long_pred, alpha = alpha, return_conf_int=True)
pred_trabajadores_1 = pd.DataFrame(pred_int, columns=['lower', 'upper'])
pred_trabajadores_1['pred'] = pred.reset_index(drop = True)
pred_trabajadores_1['ds'] = ds

# ------------------------------- 2.3 XGBOOST -------------------------------


# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "eval_metric" : ['mape'],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.001, 0.1, 0.2],
    "n_estimators": [20, 50, 100, 150],
    "colsample_bytree": [0.7, 1.0],
}

# Calculamos las características a usar
caracteristicas_trabajadores = pd.DataFrame({
    'month' : trabajadores['ds'].dt.month,
    'year' : trabajadores['ds'].dt.year,
    "promedio_3_meses" : trabajadores["y"].shift(1).rolling(window=3).mean(),
    "desvio_3_meses" : trabajadores["y"].shift(1).rolling(window=3).std(),
    "lag_1" : trabajadores["y"].shift(1),
    "lag_2" : trabajadores["y"].shift(2),
    "lag_12" : trabajadores["y"].shift(12),
})

# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
resultados_2_xgb3 = Tuner(forecaster_fun= 'XGBoost', datos=trabajadores.head(len(trabajadores)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_trabajadores.head(len(trabajadores)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['XGBoost', long_pred, resultados_2_xgb3['mape'], resultados_2_xgb3['score'], resultados_2_xgb3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_2_xgb6 = Tuner(forecaster_fun= 'XGBoost', datos=trabajadores.head(len(trabajadores)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_trabajadores.head(len(trabajadores)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['XGBoost', long_pred, resultados_2_xgb6['mape'], resultados_2_xgb6['score'], resultados_2_xgb6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_2_xgb = Tuner(forecaster_fun= 'XGBoost', datos=trabajadores, parametros=params, caracteristicas=caracteristicas_trabajadores, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['XGBoost', long_pred, resultados_2_xgb['mape'], resultados_2_xgb['score'], resultados_2_xgb['tiempo']]


# ------------------------------- 2.4 LIGHTGBM -------------------------------


# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "verbose": [-1],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.001, 0.1, 0.2],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [2,4,8,16],
    "colsample_bytree": [0.7, 1.0],
}

# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
resultados_2_lgbm3 = Tuner(forecaster_fun= 'LightGBM', datos=trabajadores.head(len(trabajadores)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_trabajadores.head(len(trabajadores)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LightGBM', long_pred, resultados_2_lgbm3['mape'], resultados_2_lgbm3['score'], resultados_2_lgbm3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_2_lgbm6 = Tuner(forecaster_fun= 'LightGBM', datos=trabajadores.head(len(trabajadores)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_trabajadores.head(len(trabajadores)-(12-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LightGBM', long_pred, resultados_2_lgbm6['mape'], resultados_2_lgbm6['score'], resultados_2_lgbm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 12
resultados_2_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=trabajadores, parametros=params, caracteristicas=caracteristicas_trabajadores, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LightGBM', long_pred, resultados_2_lgbm['mape'], resultados_2_lgbm['score'], resultados_2_lgbm['tiempo']]


# ------------------------------- 2.5 LSTM -------------------------------

# Definimos los parametros a tunear
parametros = {
    'lags' : [1, 12, 24],
    'random_seed' : [seed],
    'epochs' : [300],
    'early_stop_patience' : [10],
    'units' : [[24,42],[32],[12,24]],
    'dropout' : [0.1, 0.3],
    'activation' : ['relu', 'tanh'],
    'learning_rate' : [0.001]
    }

# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
resultados_2_lstm3 = Tuner(forecaster_fun= 'LSTM', datos=trabajadores.head(len(trabajadores)-(12-long_pred)), parametros=parametros, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LSTM', long_pred, resultados_2_lstm3['mape'], resultados_2_lstm3['score'], resultados_2_lstm3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_2_lstm6 = Tuner(forecaster_fun= 'LSTM', datos=trabajadores.head(len(trabajadores)-(12-long_pred)), parametros=parametros, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LSTM', long_pred, resultados_2_lstm6['mape'], resultados_2_lstm6['score'], resultados_2_lstm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 12
resultados_2_lstm = Tuner(forecaster_fun= 'LSTM', datos=trabajadores, parametros= parametros, alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LSTM', long_pred, resultados_2_lstm['mape'], resultados_2_lstm['score'], resultados_2_lstm['tiempo']]



# ------------------------------- 2.6 TIMEGPT -------------------------------

# Tuneamos los parametros y ajustamos el modelo
resultados_2_gpt3 = Tuner(forecaster_fun= 'TimeGPT', datos=trabajadores, alpha= alpha, long_pred = 3)
resultados_2_gpt6 = Tuner(forecaster_fun= 'TimeGPT', datos=trabajadores, alpha= alpha, long_pred = 6)
resultados_2_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=trabajadores, alpha= alpha, long_pred = 12)

# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['TimeGPT', 3, resultados_2_gpt3['mape'], resultados_2_gpt3['score'], resultados_2_gpt3['tiempo']]
metricas_2.loc[len(metricas_2)] = ['TimeGPT', 6, resultados_2_gpt6['mape'], resultados_2_gpt6['score'], resultados_2_gpt6['tiempo']]
metricas_2.loc[len(metricas_2)] = ['TimeGPT', 12, resultados_2_gpt['mape'], resultados_2_gpt['score'], resultados_2_gpt['tiempo']]




# -------------------------------------------------------------------------
# ------------------------------- SERIE 3 ---------------------------------
# -------------------------------------------------------------------------


# ------------------------------- 3.1 CARGA DE DATOS -------------------------------
import glob

# Cargamos todos los archivos txt
ruta = glob.glob('Datos/Datos meteorologicos/*.txt')
tiempo_region = pd.concat([pd.read_fwf(f, skiprows=[1], dtype={'FECHA' : str, 'HORA': str} , encoding='cp1252') for f in ruta], ignore_index=True)

# Filtramos los datos de rosario
tiempo_rosario = tiempo_region[tiempo_region['NOMBRE'] == 'ROSARIO AERO']

# Creamos la columna fecha y hora
tiempo_rosario['HORA'] = tiempo_rosario['HORA'].apply(lambda x: '0' + x if len(x) != 2 else x)
tiempo_rosario.loc[:,'ds'] = tiempo_rosario['FECHA'].apply(lambda x: x[0:2] + '-' + x[2:4] + '-' + x[4:len(x)])
tiempo_rosario['ds'] = pd.to_datetime(tiempo_rosario['ds'] + ' ' + tiempo_rosario['HORA'], format='%d-%m-%Y %H')

# Nos quedamos con las columnas utiles y renombramos la respuesta
tiempo_rosario = tiempo_rosario[['ds', 'TEMP', 'HUM', 'PNM']]
tiempo_rosario.columns = ['ds', 'y', 'HUM', 'PNM'] # % de Humedad y Presion a nivel del mar en hectopascales 


# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2
long_pred = 24 # un dia

# Creamos un dataframe donde vamos a guardar todos los resultados
metricas_3 = pd.DataFrame(columns=(['Modelo', 'Horizonte', 'MAPE', 'Interval Score', 'Tiempo']))

# ------------------------------- 3.2 ARIMA -------------------------------



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
resultados_3_arima = Tuner(forecaster_fun= 'ARIMA', datos=tiempo_rosario[['ds','y']], parametros=params, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])

# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['ARIMA', long_pred, resultados_3_arima['mape'], resultados_3_arima['score'], resultados_3_arima['tiempo']]


# Modelos manuales
temperatura_trunc = tiempo_rosario.head(len(tiempo_rosario)-long_pred).copy()
ds = tiempo_rosario.tail(long_pred)['ds'].reset_index(drop = True)

# Modelo 1
arima_temperatura_1 = ARIMA( 
    order=(1,1,1), 
    seasonal_order=(1,0,1,24)
    )

arima_temperatura_1 = arima_temperatura_1.fit(temperatura_trunc['y'], temperatura_trunc[['HUM', 'PNM']])

# Modelo 2
arima_temperatura_2 = ARIMA( 
    order=(1,1,0), 
    seasonal_order=(2,0,1,24))

arima_temperatura_2 = arima_temperatura_2.fit(temperatura_trunc['y'], temperatura_trunc[['HUM', 'PNM']])

# Modelo 3
arima_temperatura_3 = ARIMA( 
    order=(1,1,0), 
    seasonal_order=(1,1,0,24))

arima_temperatura_3 = arima_temperatura_3.fit(temperatura_trunc['y'], temperatura_trunc[['HUM', 'PNM']])

pred, pred_int = arima_temperatura_3.predict(n_periods = long_pred, X= tiempo_rosario.tail(long_pred)[['HUM', 'PNM']], alpha = alpha, return_conf_int=True)
pred_temperatura_3 = pd.DataFrame(pred_int, columns=['lower', 'upper'])
pred_temperatura_3['pred'] = pred.reset_index(drop = True)
pred_temperatura_3['ds'] = ds

# ------------------------------- 3.3 XGBOOST -------------------------------

# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "eval_metric" : ['mape'],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.001, 0.1, 0.2],
    "n_estimators": [20, 50, 100, 150],
    "colsample_bytree": [0.7, 1.0],
}
# Calculamos las características a usar
caracteristicas_temperatura = pd.DataFrame({
    'day' : tiempo_rosario['ds'].dt.day,
    'hour' : tiempo_rosario['ds'].dt.hour,
    "promedio_3_horas" : tiempo_rosario["y"].shift(1).rolling(window=3).mean(),
    "desvio_3_horas" : tiempo_rosario["y"].shift(1).rolling(window=3).std(),
    "lag_1" : tiempo_rosario["y"].shift(1),
    "lag_2" : tiempo_rosario["y"].shift(2),
    "lag_6" : tiempo_rosario["y"].shift(6),
    "lag_12" : tiempo_rosario["y"].shift(12),
    "lag_24" : tiempo_rosario["y"].shift(24),
})

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_3_xgb6 = Tuner(forecaster_fun= 'XGBoost', datos=tiempo_rosario[['ds','y']].head(len(tiempo_rosario)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_temperatura.head(len(tiempo_rosario)-(24-long_pred)), exog=tiempo_rosario[['HUM','PNM']].head(len(tiempo_rosario)-(24-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['XGBoost', long_pred, resultados_3_xgb6['mape'], resultados_3_xgb6['score'], resultados_3_xgb6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_3_xgb12 = Tuner(forecaster_fun= 'XGBoost', datos=tiempo_rosario[['ds','y']].head(len(tiempo_rosario)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_temperatura.head(len(tiempo_rosario)-(24-long_pred)), exog=tiempo_rosario[['HUM','PNM']].head(len(tiempo_rosario)-(24-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['XGBoost', long_pred, resultados_3_xgb12['mape'], resultados_3_xgb12['score'], resultados_3_xgb12['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 24
resultados_3_xgb = Tuner(forecaster_fun= 'XGBoost', datos=tiempo_rosario[['ds','y']], parametros=params, caracteristicas=caracteristicas_temperatura, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['XGBoost', long_pred, resultados_3_xgb['mape'], resultados_3_xgb['score'], resultados_3_xgb['tiempo']]


# ------------------------------- 3.4 LIGHTGBM -------------------------------


# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "verbose": [-1],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.001, 0.1, 0.2],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [2,4,8,16],
    "colsample_bytree": [0.7, 1.0],
}

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_3_lgbm6 = Tuner(forecaster_fun= 'LightGBM', datos=tiempo_rosario[['ds','y']].head(len(tiempo_rosario)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_temperatura.head(len(tiempo_rosario)-(24-long_pred)), exog=tiempo_rosario[['HUM','PNM']].head(len(tiempo_rosario)-(24-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LightGBM', long_pred, resultados_3_lgbm6['mape'], resultados_3_lgbm6['score'], resultados_3_lgbm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_3_lgbm12 = Tuner(forecaster_fun= 'LightGBM', datos=tiempo_rosario[['ds','y']].head(len(tiempo_rosario)-(12-long_pred)), parametros=params, caracteristicas=caracteristicas_temperatura.head(len(tiempo_rosario)-(24-long_pred)), exog=tiempo_rosario[['HUM','PNM']].head(len(tiempo_rosario)-(24-long_pred)), alpha= alpha, long_pred = long_pred)
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LightGBM', long_pred, resultados_3_lgbm12['mape'], resultados_3_lgbm12['score'], resultados_3_lgbm12['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 24
resultados_3_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=tiempo_rosario[['ds','y']], parametros=params, caracteristicas=caracteristicas_temperatura, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LightGBM', long_pred, resultados_3_lgbm['mape'], resultados_3_lgbm['score'], resultados_3_lgbm['tiempo']]


# ------------------------------- 3.5 LSTM -------------------------------


# Definimos los parametros a tunear
parametros = {
    'lags' : [1, 12, 24],
    'random_seed' : [seed],
    'epochs' : [300],
    'early_stop_patience' : [10],
    'units' : [[24,42],[32],[12,24]],
    'dropout' : [0.1, 0.3],
    'activation' : ['relu', 'tanh'],
    'learning_rate' : [0.001]
    }

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
resultados_3_lstm6 = Tuner(forecaster_fun= 'LSTM', datos=tiempo_rosario[['ds','y']].head(len(tiempo_rosario)-(24-long_pred)), parametros=parametros, exog=tiempo_rosario[['HUM','PNM']].head(len(tiempo_rosario)-(24-long_pred)), alpha= alpha, long_pred = long_pred, freq='h')
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LSTM', long_pred, resultados_3_lstm6['mape'], resultados_3_lstm6['score'], resultados_3_lstm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
resultados_3_lstm12 = Tuner(forecaster_fun= 'LSTM', datos=tiempo_rosario[['ds','y']].head(len(tiempo_rosario)-(24-long_pred)), parametros=parametros, exog=tiempo_rosario[['HUM','PNM']].head(len(tiempo_rosario)-(24-long_pred)), alpha= alpha, long_pred = long_pred, freq='h')
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LSTM', long_pred, resultados_3_lstm12['mape'], resultados_3_lstm12['score'], resultados_3_lstm12['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 24
resultados_3_lstm = Tuner(forecaster_fun= 'LSTM', datos=tiempo_rosario[['ds','y']], parametros= parametros, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']], freq='h')
# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LSTM', long_pred, resultados_3_lstm['mape'], resultados_3_lstm['score'], resultados_3_lstm['tiempo']]


# ------------------------------- 3.6 TIMEGPT -------------------------------


# Tuneamos los parametros y ajustamos el modelo
resultados_3_gpt6 = Tuner(forecaster_fun= 'TimeGPT', datos=tiempo_rosario[['ds','y']], alpha= alpha, long_pred = 6, exog=tiempo_rosario[['HUM','PNM']], freq='h')
resultados_3_gpt12 = Tuner(forecaster_fun= 'TimeGPT', datos=tiempo_rosario[['ds','y']], alpha= alpha, long_pred = 12, exog=tiempo_rosario[['HUM','PNM']], freq='h')
resultados_3_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=tiempo_rosario[['ds','y']], alpha= alpha, long_pred = 24, exog=tiempo_rosario[['HUM','PNM']], freq='h')

# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['TimeGPT', 6, resultados_3_gpt6['mape'], resultados_3_gpt6['score'], resultados_3_gpt6['tiempo']]
metricas_3.loc[len(metricas_3)] = ['TimeGPT', 12, resultados_3_gpt12['mape'], resultados_3_gpt12['score'], resultados_3_gpt12['tiempo']]
metricas_3.loc[len(metricas_3)] = ['TimeGPT', 24, resultados_3_gpt['mape'], resultados_3_gpt['score'], resultados_3_gpt['tiempo']]

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

resultados_arima = {
    # Residuos
    'resid_arima_atenciones_1' : arima_atenciones_1.resid(),
    'resid_arima_atenciones_2' : arima_atenciones_2.resid(),
    'resid_arima_trabajadores_1' : arima_trabajadores_1.resid(),
    'resid_arima_temperatura_1' : arima_temperatura_1.resid(),
    'resid_arima_temperatura_2' : arima_temperatura_2.resid(),
    'resid_arima_temperatura_3' : arima_temperatura_3.resid(),
    'resid_arima_atenciones_auto' : resultados_1_arima['modelo'].resid(),
    'resid_arima_trabajadores_auto' : resultados_2_arima['modelo'].resid(),
    'resid_arima_temperatura_auto' : resultados_3_arima['modelo'].resid(),
    # Pronosticos
    'pred_atenciones_1': pred_atenciones_1,
    'pred_atenciones_2': pred_atenciones_2,
    'pred_trabajadores_1': pred_trabajadores_1,
    'pred_temperatura_3': pred_temperatura_3,
    # Salidas
    'salida_arima_atenciones_1' : arima_atenciones_1.summary(),
    'salida_arima_atenciones_2' : arima_atenciones_2.summary(),
    'salida_arima_trabajadores_1' : arima_trabajadores_1.summary(),
    'salida_arima_temperatura_1' : arima_temperatura_1.summary(),
    'salida_arima_temperatura_2' : arima_temperatura_2.summary(),
    'salida_arima_temperatura_3' : arima_temperatura_3.summary(),
    'salida_arima_atenciones_auto' : resultados_1_arima['modelo'].summary(),
    'salida_arima_trabajadores_auto' : resultados_2_arima['modelo'].summary(),
    'salida_arima_temperatura_auto' : resultados_3_arima['modelo'].summary(),
}

# Guardamos el ambiente

save_env(env_dict= {
    "resultados_1_arima" : {k: v for k, v in resultados_1_arima.items() if k != 'modelo'},
    "resultados_1_xgb" : {k: v for k, v in resultados_1_xgb.items() if k != 'modelo'},
    "resultados_1_xgb3" : {k: v for k, v in resultados_1_xgb3.items() if k != 'modelo'},
    "resultados_1_xgb6" : {k: v for k, v in resultados_1_xgb6.items() if k != 'modelo'},
    "resultados_1_lgbm" : {k: v for k, v in resultados_1_lgbm.items() if k != 'modelo'},
    "resultados_1_lgbm3" : {k: v for k, v in resultados_1_lgbm3.items() if k != 'modelo'},
    "resultados_1_lgbm6" : {k: v for k, v in resultados_1_lgbm6.items() if k != 'modelo'},
    "resultados_1_lstm" : {k: v for k, v in resultados_1_lstm.items() if k != 'modelo'},
    "resultados_1_lstm3" : {k: v for k, v in resultados_1_lstm3.items() if k != 'modelo'},
    "resultados_1_lstm6" : {k: v for k, v in resultados_1_lstm6.items() if k != 'modelo'},
    "resultados_1_gpt" : {k: v for k, v in resultados_1_gpt.items() if k != 'modelo'},
    "resultados_1_gpt3" : {k: v for k, v in resultados_1_gpt3.items() if k != 'modelo'},
    "resultados_1_gpt6" : {k: v for k, v in resultados_1_gpt6.items() if k != 'modelo'},
    "metricas_1" : metricas_1,
    "resultados_2_arima" : {k: v for k, v in resultados_2_arima.items() if k != 'modelo'},
    "resultados_2_xgb" : {k: v for k, v in resultados_2_xgb.items() if k != 'modelo'},
    "resultados_2_xgb3" : {k: v for k, v in resultados_2_xgb3.items() if k != 'modelo'},
    "resultados_2_xgb6" : {k: v for k, v in resultados_2_xgb6.items() if k != 'modelo'},
    "resultados_2_lgbm" : {k: v for k, v in resultados_2_lgbm.items() if k != 'modelo'},
    "resultados_2_lgbm3" : {k: v for k, v in resultados_2_lgbm3.items() if k != 'modelo'},
    "resultados_2_lgbm6" : {k: v for k, v in resultados_2_lgbm6.items() if k != 'modelo'},
    "resultados_2_lstm" : {k: v for k, v in resultados_2_lstm.items() if k != 'modelo'},
    "resultados_2_lstm3" : {k: v for k, v in resultados_2_lstm3.items() if k != 'modelo'},
    "resultados_2_lstm6" : {k: v for k, v in resultados_2_lstm6.items() if k != 'modelo'},
    "resultados_2_gpt" : {k: v for k, v in resultados_2_gpt.items() if k != 'modelo'},
    "resultados_2_gpt3" : {k: v for k, v in resultados_2_gpt3.items() if k != 'modelo'},
    "resultados_2_gpt6" : {k: v for k, v in resultados_2_gpt6.items() if k != 'modelo'},
    "metricas_2" : metricas_2,
    "resultados_3_arima" : {k: v for k, v in resultados_3_arima.items() if k != 'modelo'},
    "resultados_3_xgb" : {k: v for k, v in resultados_3_xgb.items() if k != 'modelo'},
    "resultados_3_xgb6" : {k: v for k, v in resultados_3_xgb6.items() if k != 'modelo'},
    "resultados_3_xgb12" : {k: v for k, v in resultados_3_xgb12.items() if k != 'modelo'},
    "resultados_3_lgbm" : {k: v for k, v in resultados_3_lgbm.items() if k != 'modelo'},
    "resultados_3_lgbm6" : {k: v for k, v in resultados_3_lgbm6.items() if k != 'modelo'},
    "resultados_3_lgbm12" : {k: v for k, v in resultados_3_lgbm12.items() if k != 'modelo'},
    "resultados_3_lstm" : {k: v for k, v in resultados_3_lstm.items() if k != 'modelo'},
    "resultados_3_lstm6" : {k: v for k, v in resultados_3_lstm6.items() if k != 'modelo'},
    "resultados_3_lstm12" : {k: v for k, v in resultados_3_lstm12.items() if k != 'modelo'},
    "resultados_3_gpt" : {k: v for k, v in resultados_3_gpt.items() if k != 'modelo'},
    "resultados_3_gpt6" : {k: v for k, v in resultados_3_gpt6.items() if k != 'modelo'},
    "resultados_3_gpt12" : {k: v for k, v in resultados_3_gpt12.items() if k != 'modelo'},
    "metricas_3" : metricas_3,
    'resultados_arima' : resultados_arima,
}, filename="Codigo/Ambiente/Amb_Aplicacion.pkl")

# Guardamos los modelos

save_env(env_dict= {
    'arima_atenciones_auto': resultados_1_arima['modelo'],
    "modelo_1_xgb" : resultados_1_xgb['modelo'],
    "modelo_1_lgbm" : resultados_1_lgbm['modelo'],
    "modelo_1_lstm" : resultados_1_lstm['modelo'],
    'arima_trabajadores_auto':resultados_2_arima['modelo'],
    "modelo_2_xgb" : resultados_2_xgb['modelo'],
    "modelo_2_lgbm" : resultados_2_lgbm['modelo'],
    "modelo_2_lstm" : resultados_2_lstm['modelo'],
    'arima_temperatura_auto':resultados_3_arima['modelo'],
    "modelo_3_xgb" : resultados_3_xgb['modelo'],
    "modelo_3_lgbm" : resultados_3_lgbm['modelo'],
    "modelo_3_lstm" : resultados_3_lstm['modelo'],

    'arima_atenciones_1': arima_atenciones_1,
    'arima_atenciones_2': arima_atenciones_2,
    'arima_trabajadores_1': arima_trabajadores_1,
    'arima_temperatura_1': arima_temperatura_1,
    'arima_temperatura_2': arima_temperatura_2,
    'arima_temperatura_3': arima_temperatura_3,
}, filename="Codigo/Ambiente/modelos_aplicacion.pkl")




# ---------------- PRUEBA --------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------

from scalecast.Pipeline import Transformer, Reverter, Pipeline
from scalecast.util import backtest_for_resid_matrix, get_backtest_resid_matrix, overwrite_forecast_intervals
from scalecast.Forecaster import Forecaster
import tensorflow.keras.backend as K
import gc
from tensorflow.keras.callbacks import EarlyStopping

f = Forecaster(
    y = tiempo_rosario.head(len(tiempo_rosario)- long_pred)['y'],
    current_dates = tiempo_rosario.head(len(tiempo_rosario)- long_pred)['ds'],
    future_dates=24,
    freq='h',
    test_length = 24,
    cis = False,
    metrics = ['mape'],
    carry_fit_models =False
)
f.add_series(tiempo_rosario['HUM'], called = 'HUM', first_date=tiempo_rosario['ds'].head(1).values[0], forward_pad = False, back_pad=False)
f.add_series(tiempo_rosario['PNM'], called = 'PNM', first_date=tiempo_rosario['ds'].head(1).values[0], forward_pad = False, back_pad=False)
f.eval_cis(mode=True, cilevel=0.8)

early_stop = EarlyStopping(
    monitor='loss',       
    patience=10,          
    restore_best_weights=True
)

transformer = Transformer(
    transformers = [
        ('DetrendTransform',{'poly_order':2}),
        'DeseasonTransform',
    ],
)
reverter = Reverter(
    reverters = [
        'DeseasonRevert',
        'DetrendRevert',
    ],
    base_transformer = transformer,
)



def forecaster(f):
    f.set_estimator('rnn')
    f.manual_forecast(
        lags = 20,
        layers_struct = [
            (['LSTM',{'units':36,'activation':'relu'}]),
        ],
        epochs=200,
        call_me = 'lstm',
        verbose=True,
        random_seed = seed,
        learning_rate  =0.001,
        callbacks=[early_stop]
    )


pipeline = Pipeline(
    steps = [
        ('Transform',transformer),
        ('Forecast',forecaster),
        ('Revert',reverter),
    ]
)

f = pipeline.fit_predict(f)

# backtest_results = backtest_for_resid_matrix(
#     f,
#     pipeline=pipeline,
#     alpha = .2,
#     jump_back = 3
# )
# backtest_resid_matrix = get_backtest_resid_matrix(backtest_results)
# overwrite_forecast_intervals(f,backtest_resid_matrix=backtest_resid_matrix,alpha=.2)

forecaster = f.export(cis = True, dfs ='lvl_fcsts')
forecaster.columns = ['ds', 'pred', 'upper', 'lower']
plot_forecast(tiempo_rosario, forecaster, long = 72)
forecaster

f.drop_all_Xvars()
K.clear_session()
gc.collect()

# -----------------------------

# Para el manejo de estructuras de datos
import pandas as pd

# Para ajustar los modelos arima manualmente
from pmdarima.arima import ARIMA

# Cargamos funciones
from Codigo.Funciones import save_env, load_env
from Codigo.tuner_fun import Tuner

# Definimos una semilla
seed = 11072001


# ------------------------------- 3.1 CARGA DE DATOS -------------------------------
import glob

# Cargamos todos los archivos txt
ruta = glob.glob('Datos/Datos meteorologicos/*.txt')
tiempo_region = pd.concat([pd.read_fwf(f, skiprows=[1], dtype={'FECHA' : str, 'HORA': str} , encoding='cp1252') for f in ruta], ignore_index=True)

# Filtramos los datos de rosario
tiempo_rosario = tiempo_region[tiempo_region['NOMBRE'] == 'ROSARIO AERO']

# Creamos la columna fecha y hora
tiempo_rosario['HORA'] = tiempo_rosario['HORA'].apply(lambda x: '0' + x if len(x) != 2 else x)
tiempo_rosario.loc[:,'ds'] = tiempo_rosario['FECHA'].apply(lambda x: x[0:2] + '-' + x[2:4] + '-' + x[4:len(x)])
tiempo_rosario['ds'] = pd.to_datetime(tiempo_rosario['ds'] + ' ' + tiempo_rosario['HORA'], format='%d-%m-%Y %H')

# Nos quedamos con las columnas utiles y renombramos la respuesta
tiempo_rosario = tiempo_rosario[['ds', 'TEMP', 'HUM', 'PNM']]
tiempo_rosario.columns = ['ds', 'y', 'HUM', 'PNM'] # % de Humedad y Presion a nivel del mar en hectopascales 


# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2
long_pred = 24 # un dia

    # f.manual_forecast(
    #     lags = 18,
    #     layers_struct = [
    #         (['LSTM',{'units':24,'activation':'relu'}] + ['LSTM',{'units':48,'activation':'relu'}]),
    #     ],
    #     epochs=200,
    #     call_me = 'lstm',
    #     verbose=False,
    #     random_seed = seed,
    # )



parametros = {
    'lags' : [12],
    'random_seed' : [seed],
    'epochs' : [300],
    'early_stop_patience' : [10],
    'units' : [[32]],
    'dropout' : [0.3],
    'activation' : ['tanh'],
    'learning_rate' : [0.001]
    }

resultados_3_lstm = Tuner(forecaster_fun= 'LSTM', datos=tiempo_rosario[['ds','y']], parametros= parametros, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']], freq='h')

from Codigo.Funciones import plot_forecast
plot_forecast(tiempo_rosario, resultados_3_lstm['pred'], long = 72)



# Guardar las metricas y resultados
metricas_3.loc[len(metricas_3)] = ['LSTM', long_pred, resultados_3_lstm['mape'], resultados_3_lstm['score'], resultados_3_lstm['tiempo']]


# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
# Guardamos las metricas
metricas_1.loc[7] = ['LSTM', long_pred, resultados_1_lstm3['mape'], resultados_1_lstm3['score'], resultados_1_lstm3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
# Guardamos las metricas
metricas_1.loc[8] = ['LSTM', long_pred, resultados_1_lstm6['mape'], resultados_1_lstm6['score'], resultados_1_lstm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
# Guardamos las metricas
metricas_1.loc[9] = ['LSTM', long_pred, resultados_1_lstm['mape'], resultados_1_lstm['score'], resultados_1_lstm['tiempo']]


# Tuneamos los parametros y ajustamos el modelo con horizonte 3
long_pred = 3
# Guardamos las metricas
metricas_2.loc[7] = ['LSTM', long_pred, resultados_2_lstm3['mape'], resultados_2_lstm3['score'], resultados_2_lstm3['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
# Guardamos las metricas
metricas_2.loc[8] = ['LSTM', long_pred, resultados_2_lstm6['mape'], resultados_2_lstm6['score'], resultados_2_lstm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 12
# Guardamos las metricas
metricas_2.loc[9] = ['LSTM', long_pred, resultados_2_lstm['mape'], resultados_2_lstm['score'], resultados_2_lstm['tiempo']]


# Tuneamos los parametros y ajustamos el modelo con horizonte 6
long_pred = 6
# Guardamos las metricas
metricas_3.loc[7] = ['LSTM', long_pred, resultados_3_lstm6['mape'], resultados_3_lstm6['score'], resultados_3_lstm6['tiempo']]

# Tuneamos los parametros y ajustamos el modelo con horizonte 12
long_pred = 12
# Guardamos las metricas
metricas_3.loc[8] = ['LSTM', long_pred, resultados_3_lstm12['mape'], resultados_3_lstm12['score'], resultados_3_lstm12['tiempo']]

# Tuneamos los parametros y ajustamos el modelo
long_pred = 24
# Guardamos las metricas
metricas_3.loc[9] = ['LSTM', long_pred, resultados_3_lstm['mape'], resultados_3_lstm['score'], resultados_3_lstm['tiempo']]



save_env(env_dict= {
    "resultados_1_arima" : resultados_1_arima, #{k: v for k, v in resultados_1_arima.items() if k != 'modelo'},
    "resultados_1_xgb" : resultados_1_xgb, #{k: v for k, v in resultados_1_xgb.items() if k != 'modelo'},
    "resultados_1_xgb3" : resultados_1_xgb3, #{k: v for k, v in resultados_1_xgb3.items() if k != 'modelo'},
    "resultados_1_xgb6" : resultados_1_xgb6, #{k: v for k, v in resultados_1_xgb6.items() if k != 'modelo'},
    "resultados_1_lgbm" : resultados_1_lgbm, #{k: v for k, v in resultados_1_lgbm.items() if k != 'modelo'},
    "resultados_1_lgbm3" : resultados_1_lgbm3, #{k: v for k, v in resultados_1_lgbm3.items() if k != 'modelo'},
    "resultados_1_lgbm6" : resultados_1_lgbm6, #{k: v for k, v in resultados_1_lgbm6.items() if k != 'modelo'},
    "resultados_1_lstm" : {k: v for k, v in resultados_1_lstm.items() if k != 'modelo'},
    "resultados_1_lstm3" : {k: v for k, v in resultados_1_lstm3.items() if k != 'modelo'},
    "resultados_1_lstm6" : {k: v for k, v in resultados_1_lstm6.items() if k != 'modelo'},
    "resultados_1_gpt" : resultados_1_gpt, #{k: v for k, v in resultados_1_gpt.items() if k != 'modelo'},
    "resultados_1_gpt3" : resultados_1_gpt3, #{k: v for k, v in resultados_1_gpt3.items() if k != 'modelo'},
    "resultados_1_gpt6" : resultados_1_gpt6, #{k: v for k, v in resultados_1_gpt6.items() if k != 'modelo'},
    "metricas_1" : metricas_1,
    "resultados_2_arima" : resultados_2_arima, #{k: v for k, v in resultados_2_arima.items() if k != 'modelo'},
    "resultados_2_xgb" : resultados_2_xgb, #{k: v for k, v in resultados_2_xgb.items() if k != 'modelo'},
    "resultados_2_xgb3" : resultados_2_xgb3, #{k: v for k, v in resultados_2_xgb3.items() if k != 'modelo'},
    "resultados_2_xgb6" : resultados_2_xgb6, #{k: v for k, v in resultados_2_xgb6.items() if k != 'modelo'},
    "resultados_2_lgbm" : resultados_2_lgbm, #{k: v for k, v in resultados_2_lgbm.items() if k != 'modelo'},
    "resultados_2_lgbm3" : resultados_2_lgbm3, #{k: v for k, v in resultados_2_lgbm3.items() if k != 'modelo'},
    "resultados_2_lgbm6" : resultados_2_lgbm6, #{k: v for k, v in resultados_2_lgbm6.items() if k != 'modelo'},
    "resultados_2_lstm" : {k: v for k, v in resultados_2_lstm.items() if k != 'modelo'},
    "resultados_2_lstm3" : {k: v for k, v in resultados_2_lstm3.items() if k != 'modelo'},
    "resultados_2_lstm6" : {k: v for k, v in resultados_2_lstm6.items() if k != 'modelo'},
    "resultados_2_gpt" : resultados_2_gpt, #{k: v for k, v in resultados_2_gpt.items() if k != 'modelo'},
    "resultados_2_gpt3" : resultados_2_gpt3, #{k: v for k, v in resultados_2_gpt3.items() if k != 'modelo'},
    "resultados_2_gpt6" : resultados_2_gpt6, #{k: v for k, v in resultados_2_gpt6.items() if k != 'modelo'},
    "metricas_2" : metricas_2,
    "resultados_3_arima" : resultados_3_arima, # {k: v for k, v in resultados_3_arima.items() if k != 'modelo'},
    "resultados_3_xgb" : resultados_3_xgb, # {k: v for k, v in resultados_3_xgb.items() if k != 'modelo'},
    "resultados_3_xgb6" : resultados_3_xgb6, # {k: v for k, v in resultados_3_xgb6.items() if k != 'modelo'},
    "resultados_3_xgb12" : resultados_3_xgb12, # {k: v for k, v in resultados_3_xgb12.items() if k != 'modelo'},
    "resultados_3_lgbm" : resultados_3_lgbm, # {k: v for k, v in resultados_3_lgbm.items() if k != 'modelo'},
    "resultados_3_lgbm6" : resultados_3_lgbm6, # {k: v for k, v in resultados_3_lgbm6.items() if k != 'modelo'},
    "resultados_3_lgbm12" : resultados_3_lgbm12, # {k: v for k, v in resultados_3_lgbm12.items() if k != 'modelo'},
    "resultados_3_lstm" : {k: v for k, v in resultados_3_lstm.items() if k != 'modelo'},
    "resultados_3_lstm6" : {k: v for k, v in resultados_3_lstm6.items() if k != 'modelo'},
    "resultados_3_lstm12" : {k: v for k, v in resultados_3_lstm12.items() if k != 'modelo'},
    "resultados_3_gpt" : resultados_3_gpt, # {k: v for k, v in resultados_3_gpt.items() if k != 'modelo'},
    "resultados_3_gpt6" : resultados_3_gpt6, # {k: v for k, v in resultados_3_gpt6.items() if k != 'modelo'},
    "resultados_3_gpt12" : resultados_3_gpt12, # {k: v for k, v in resultados_3_gpt12.items() if k != 'modelo'},
    "metricas_3" : metricas_3,
    'resultados_arima' : resultados_arima,
}, filename="Codigo/Ambiente/Amb_Aplicacion.pkl")

