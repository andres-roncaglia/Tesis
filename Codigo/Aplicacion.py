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
metricas_1 = pd.DataFrame(columns=(['Modelo', 'MAPE', 'Interval Score', 'Tiempo']))

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
metricas_1.loc[len(metricas_1)] = ['ARIMA', resultados_1_arima['mape'], resultados_1_arima['score'], resultados_1_arima['tiempo']]

# Modelos manuales
atenciones_trunc = atenciones_guardia.head(len(atenciones_guardia)-long_pred).copy()

# Modelo 1
arima_atenciones_1 = ARIMA( 
    order=(0, 1, 1), 
    seasonal_order=(0,1,0,12))

arima_atenciones_1 = arima_atenciones_1.fit(atenciones_trunc['y'])


# Modelo 2
arima_atenciones_2 = ARIMA( 
    order=(0, 1, 0), 
    seasonal_order=(0,1,1,12))

arima_atenciones_2 = arima_atenciones_2.fit(atenciones_trunc['y'])


# ------------------------------- 1.3 XGBOOST -------------------------------


# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150]

}

# Tuneamos los parametros y ajustamos el modelo
resultados_1_xgb = Tuner(forecaster_fun= 'XGBoost', datos=atenciones_guardia, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['XGBoost', resultados_1_xgb['mape'], resultados_1_xgb['score'], resultados_1_xgb['tiempo']]

# ------------------------------- 1.4 LIGHTGBM -------------------------------


# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [5, 10, 20, 30, 50]
}

# Tuneamos los parametros y ajustamos el modelo
resultados_1_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=atenciones_guardia, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LightGBM', resultados_1_lgbm['mape'], resultados_1_lgbm['score'], resultados_1_lgbm['tiempo']]


# ------------------------------- 1.5 LSTM -------------------------------

# Definimos los parametros a tunear
parametros = {
    'max_steps' : [50, 100, 200, 500],
    'random_seed' : [seed],
    'encoder_n_layers' : [1,2,3],
    'decoder_layers' : [1,2,3]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_1_lstm = Tuner(forecaster_fun= 'LSTM', datos=atenciones_guardia, parametros= parametros, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['LSTM', resultados_1_lstm['mape'], resultados_1_lstm['score'], resultados_1_lstm['tiempo']]


# ------------------------------- 1.6 TIMEGPT -------------------------------


# Definimos los parametros a tunear
parametros = {
    'finetune_loss' : ['mape'],
    'finetune_steps' : [1,2,5,10,15],
    'finetune_depth' : [1, 2, 3, 5]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_1_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=atenciones_guardia, parametros= parametros, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_1.loc[len(metricas_1)] = ['TimeGPT', resultados_1_gpt['mape'], resultados_1_gpt['score'], resultados_1_gpt['tiempo']]



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


# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2
long_pred = 12

# Creamos un dataframe donde vamos a guardar todos los resultados
metricas_2 = pd.DataFrame(columns=(['Modelo', 'MAPE', 'Interval Score', 'Tiempo']))


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
metricas_2.loc[len(metricas_2)] = ['ARIMA', resultados_2_arima['mape'], resultados_2_arima['score'], resultados_2_arima['tiempo']]

# Modelos manuales
trabajadores_trunc = trabajadores.head(len(trabajadores)-long_pred).copy()

# Modelo 1
arima_trabajadores_1 = ARIMA( 
    order=(0,1,1), 
    seasonal_order=(1,1,0,12))

arima_trabajadores_1 = arima_trabajadores_1.fit(trabajadores_trunc['y'])



# ------------------------------- 2.3 XGBOOST -------------------------------


# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150]

}

# Tuneamos los parametros y ajustamos el modelo
resultados_2_xgb = Tuner(forecaster_fun= 'XGBoost', datos=trabajadores, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['XGBoost', resultados_2_xgb['mape'], resultados_2_xgb['score'], resultados_2_xgb['tiempo']]


# ------------------------------- 2.4 LIGHTGBM -------------------------------


# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [5, 10, 20, 30, 50]
}

# Tuneamos los parametros y ajustamos el modelo
resultados_2_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=trabajadores, parametros=params, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LightGBM', resultados_2_lgbm['mape'], resultados_2_lgbm['score'], resultados_2_lgbm['tiempo']]


# ------------------------------- 2.5 LSTM -------------------------------

# Definimos los parametros a tunear
parametros = {
    'max_steps' : [50, 100, 200, 500],
    'random_seed' : [seed],
    'encoder_n_layers' : [1,2,3],
    'decoder_layers' : [1,2,3]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_2_lstm = Tuner(forecaster_fun= 'LSTM', datos=trabajadores, parametros= parametros, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['LSTM', resultados_2_lstm['mape'], resultados_2_lstm['score'], resultados_2_lstm['tiempo']]



# ------------------------------- 2.6 TIMEGPT -------------------------------


# Definimos los parametros a tunear
parametros = {
    'finetune_loss' : ['mape'],
    'finetune_steps' : [1,2,5,10,15],
    'finetune_depth' : [1, 2, 3, 5]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_2_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=trabajadores, parametros= parametros, alpha= alpha, long_pred = long_pred)

# Guardamos las metricas
metricas_2.loc[len(metricas_2)] = ['TimeGPT', resultados_2_gpt['mape'], resultados_2_gpt['score'], resultados_2_gpt['tiempo']]





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
metricas_3 = pd.DataFrame(columns=(['Modelo', 'MAPE', 'Interval Score', 'Tiempo']))

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
metricas_3.loc[len(metricas_3)] = ['ARIMA', resultados_3_arima['mape'], resultados_3_arima['score'], resultados_3_arima['tiempo']]


# Modelos manuales
temperatura_trunc = tiempo_rosario.head(len(tiempo_rosario)-long_pred).copy()

# Modelo 1
arima_temperatura_1 = ARIMA( 
    order=(1,1,1), 
    seasonal_order=(1,0,1,24))

arima_temperatura_1 = arima_temperatura_1.fit(temperatura_trunc['y'])

# Modelo 2
arima_temperatura_2 = ARIMA( 
    order=(1,1,0), 
    seasonal_order=(2,0,1,24))

arima_temperatura_2 = arima_temperatura_2.fit(temperatura_trunc['y'])

# Modelo 3
arima_temperatura_3 = ARIMA( 
    order=(1,1,0), 
    seasonal_order=(1,1,0,24))

arima_temperatura_3 = arima_temperatura_3.fit(temperatura_trunc['y'])

# ------------------------------- 3.3 XGBOOST -------------------------------



# Definimos los parametros a tunear
params = {
    "tree_method": ['exact'],
    "random_state": [seed],
    "max_leaves": [2,4,8,16],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150]

}

# Tuneamos los parametros y ajustamos el modelo
resultados_3_xgb = Tuner(forecaster_fun= 'XGBoost', datos=tiempo_rosario[['ds','y']], parametros=params, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])

# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['XGBoost', resultados_3_xgb['mape'], resultados_3_xgb['score'], resultados_3_xgb['tiempo']]


# ------------------------------- 3.4 LIGHTGBM -------------------------------


# Definimos los parametros a tunear
params = {
    "random_state": [seed],
    "max_depth": [2,3,4,5],
    "learning_rate": [0.1, 0.2, 0.3],
    "n_estimators": [20, 50, 100, 150],
    'num_leaves' : [5, 10, 20, 30, 50]
}

# Tuneamos los parametros y ajustamos el modelo
resultados_3_lgbm = Tuner(forecaster_fun= 'LightGBM', datos=tiempo_rosario[['ds','y']], parametros=params, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])

# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LightGBM', resultados_3_lgbm['mape'], resultados_3_lgbm['score'], resultados_3_lgbm['tiempo']]



# ------------------------------- 3.5 LSTM -------------------------------


# Definimos los parametros a tunear
parametros = {
    'max_steps' : [50, 100, 200, 500],
    'random_seed' : [seed],
    'encoder_n_layers' : [1,2,3],
    'decoder_layers' : [1,2,3]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_3_lstm = Tuner(forecaster_fun= 'LSTM', datos=tiempo_rosario[['ds','y']], parametros= parametros, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])

# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['LSTM', resultados_3_lstm['mape'], resultados_3_lstm['score'], resultados_3_lstm['tiempo']]


# ------------------------------- 3.6 TIMEGPT -------------------------------



# Definimos los parametros a tunear
parametros = {
    'finetune_loss' : ['mape'],
    'finetune_steps' : [1,2,5,10,15],
    'finetune_depth' : [1, 2, 3, 5]
    }

# Tuneamos los parametros y ajustamos el modelo
resultados_3_gpt = Tuner(forecaster_fun= 'TimeGPT', datos=tiempo_rosario[['ds','y']], parametros= parametros, alpha= alpha, long_pred = long_pred, exog=tiempo_rosario[['HUM','PNM']])

# Guardamos las metricas
metricas_3.loc[len(metricas_3)] = ['TimeGPT', resultados_3_gpt['mape'], resultados_3_gpt['score'], resultados_3_gpt['tiempo']]



# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

modelos_arima = {
    'arima_atenciones_1': arima_atenciones_1,
    'arima_atenciones_2': arima_atenciones_2,
    'arima_trabajadores_1': arima_trabajadores_1,
    'arima_temperatura_1': arima_temperatura_1,
    'arima_temperatura_2': arima_temperatura_2,
    'arima_temperatura_3': arima_temperatura_3,
    'arima_atenciones_auto': resultados_1_arima['modelo'],
    'arima_trabajadores_auto':resultados_2_arima['modelo'],
    'arima_temperatura_auto':resultados_3_arima['modelo']
}

# Guardamos el ambiente

save_env(env_dict= {
    "resultados_1_arima" : {k: v for k, v in resultados_1_arima.items() if k != 'modelo'},
    "resultados_1_xgb" : {k: v for k, v in resultados_1_xgb.items() if k != 'modelo'},
    "resultados_1_lgbm" : {k: v for k, v in resultados_1_lgbm.items() if k != 'modelo'},
    "resultados_1_lstm" : {k: v for k, v in resultados_1_lstm.items() if k != 'modelo'},
    #"resultados_1_gpt" : {k: v for k, v in resultados_1_gpt.items() if k != 'modelo'},
    "metricas_1" : metricas_1,
    "resultados_2_arima" : {k: v for k, v in resultados_2_arima.items() if k != 'modelo'},
    "resultados_2_xgb" : {k: v for k, v in resultados_2_xgb.items() if k != 'modelo'},
    "resultados_2_lgbm" : {k: v for k, v in resultados_2_lgbm.items() if k != 'modelo'},
    "resultados_2_lstm" : {k: v for k, v in resultados_2_lstm.items() if k != 'modelo'},
    #"resultados_2_gpt" : {k: v for k, v in resultados_2_gpt.items() if k != 'modelo'},
    "metricas_2" : metricas_2,
    "resultados_3_arima" : {k: v for k, v in resultados_3_arima.items() if k != 'modelo'},
    "resultados_3_xgb" : {k: v for k, v in resultados_3_xgb.items() if k != 'modelo'},
    "resultados_3_lgbm" : {k: v for k, v in resultados_3_lgbm.items() if k != 'modelo'},
    "resultados_3_lstm" : {k: v for k, v in resultados_3_lstm.items() if k != 'modelo'},
    #"resultados_3_gpt" : {k: v for k, v in resultados_3_gpt.items() if k != 'modelo'},
    "metricas_3" : metricas_3
}, filename="Codigo/Ambiente/Amb_Aplicacion.pkl")

# Guardamos los modelos

save_env(env_dict= {
    "modelo_1_xgb" : resultados_1_xgb['modelo'],
    "modelo_1_lgbm" : resultados_1_lgbm['modelo'],
    "modelo_1_lstm" : resultados_1_lstm['modelo'],
    #"modelo_1_gpt" : resultados_1_gpt['modelo'],
    "modelo_2_xgb" : resultados_2_xgb['modelo'],
    "modelo_2_lgbm" : resultados_2_lgbm['modelo'],
    "modelo_2_lstm" : resultados_2_lstm['modelo'],
    #"modelo_2_gpt" : resultados_2_gpt['modelo'],
    "modelo_3_xgb" : resultados_3_xgb['modelo'],
    "modelo_3_lgbm" : resultados_3_lgbm['modelo'],
    "modelo_3_lstm" : resultados_3_lstm['modelo'],
    #"modelo_3_gpt" : resultados_3_gpt['modelo'],
    "modelos_arima":modelos_arima
}, filename="Codigo/Ambiente/modelos_aplicacion.pkl")