# ------------------------------------- LIBRERIAS -------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd

# Para calcular el error medio cuadratico
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# Funciones utiles
import sys
sys.path.append('Codigo')
from Funciones import interval_score, save_env, load_env


# Para Chronos
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


# Definimos una semilla
seed = 11072001


# Cargamos el ambiente
globals().update(load_env('Codigo/Ambiente/Amb_Aplicacion_chronos.pkl'))
globals().update(load_env('Codigo/Ambiente/Modelos_chronos.pkl'))

# Crear una metrica propia
# https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html



# ------------------------------- 1.1 CARGA DE DATOS -------------------------------

atenciones_guardia = pd.read_excel(io='Datos/Atenciones de guardia en el HNVV por patologías respiratorias (vigiladas por epidemiología).xlsx' )

# Aseguro que la columna fecha tenga el formato adecuado
atenciones_guardia['fec'] = pd.to_datetime(atenciones_guardia['fec'], format='%Y-%m-%d')

# Filtro las columnas importantes y las renombro
atenciones_guardia = atenciones_guardia[['fec', 'frec']]
atenciones_guardia.columns = ['timestamp', 'target']

# Agrego una columna identificadora necesaria para Chronos
atenciones_guardia['item_id'] = 0

# Definicion del nivel de significacion
alpha = 0.2

q_lower = alpha/2
q_upper = 1-alpha/2


# ------------------------------- 1.2 AJUSTE DEL MODELO CON HORIZONTE 12 -------------------------------
# Largo del pronostico
long_pred = 12

# Divido en conjunto entrenamiento y prueba
atenciones_guardia_train = atenciones_guardia.head(atenciones_guardia.shape[0]-long_pred)
atenciones_guardia_test = atenciones_guardia.tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
atenciones_guardia_train = TimeSeriesDataFrame(atenciones_guardia_train)

# Definimos y ajustamos el modelo
modelo_1 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    ).fit(
    atenciones_guardia_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_1.predict(atenciones_guardia_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(atenciones_guardia_test['target'], predictions['pred'])
score = interval_score(obs=atenciones_guardia_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

atenciones_guardia_test.drop('item_id', axis = 1, inplace = True)
atenciones_guardia.columns = ['ds', 'y']
atenciones_guardia_test.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
atenciones_guardia_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : atenciones_guardia_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_1.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_1_chronos = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}

# ------------------------------- 1.2 AJUSTE DEL MODELO CON HORIZONTE 3 -------------------------------
# Largo del pronostico
long_pred = 3

# Divido en conjunto entrenamiento y prueba
atenciones_guardia_train = atenciones_guardia.head(atenciones_guardia.shape[0]-12)
atenciones_guardia_test = (atenciones_guardia.head(atenciones_guardia.shape[0]-(12-long_pred))).tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
atenciones_guardia_train = TimeSeriesDataFrame(atenciones_guardia_train)

# Definimos y ajustamos el modelo
modelo_1_3 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    ).fit(
    atenciones_guardia_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_1_3.predict(atenciones_guardia_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(atenciones_guardia_test['target'], predictions['pred'])
score = interval_score(obs=atenciones_guardia_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

atenciones_guardia_test.drop('item_id', axis = 1, inplace = True)
atenciones_guardia.columns = ['ds', 'y']
atenciones_guardia_test.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
atenciones_guardia_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : atenciones_guardia_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_1_3.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_1_chronos3 = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}

# ------------------------------- 1.2 AJUSTE DEL MODELO CON HORIZONTE 6 -------------------------------
# Largo del pronostico
long_pred = 6

# Divido en conjunto entrenamiento y prueba
atenciones_guardia_train = atenciones_guardia.head(atenciones_guardia.shape[0]-12)
atenciones_guardia_test = (atenciones_guardia.head(atenciones_guardia.shape[0]-(12-long_pred))).tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
atenciones_guardia_train = TimeSeriesDataFrame(atenciones_guardia_train)

# Definimos y ajustamos el modelo
modelo_1_6 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    ).fit(
    atenciones_guardia_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_1_6.predict(atenciones_guardia_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(atenciones_guardia_test['target'], predictions['pred'])
score = interval_score(obs=atenciones_guardia_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

atenciones_guardia_test.drop('item_id', axis = 1, inplace = True)
atenciones_guardia.columns = ['ds', 'y']
atenciones_guardia_test.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
atenciones_guardia_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : atenciones_guardia_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_1_6.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_1_chronos6 = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}

# -------------------------------------------------------------------------
# ------------------------------- SERIE 2 ---------------------------------
# -------------------------------------------------------------------------

# ------------------------------- 2.1 CARGA DE DATOS -------------------------------


# Cargamos los datos
trabajadores = pd.read_excel(io='Datos/trabajoregistrado_2502_estadisticas.xlsx', sheet_name= 'A.2.1', thousands='.', decimal=',', header=1, usecols='A,M', skipfooter=5, skiprows=84)

# Renombramos las columnas
trabajadores.columns = ['timestamp', 'target']

# Agrego una columna identificadora necesaria para Chronos
trabajadores['item_id'] = 0

# Asignamos formato fecha
meses = {
    'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 
    'may': '05', 'jun': '06', 'jul': '07', 'ago': '08', 
    'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
}

trabajadores['timestamp'] = trabajadores['timestamp'].str.replace('*','')
trabajadores['timestamp'] = trabajadores['timestamp'].apply(
    lambda x: '01-' + x.replace(x.split('-')[0], meses.get(x.split('-')[0].lower(), '')).replace(x.split('-')[1], '20' + x.split('-')[1])
)

trabajadores['timestamp'] = pd.to_datetime(trabajadores['timestamp'], format='%d-%m-%Y')

# Eliminamos los datos del 2025 para tener solo años completos
trabajadores = trabajadores[trabajadores['timestamp'].dt.year != 2025]

# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2

q_lower = alpha/2
q_upper = 1-alpha/2

# ------------------------------- 2.2 AJUSTE DEL MODELO -------------------------------
# Largo del pronostico
long_pred = 12

# Divido en conjunto entrenamiento y prueba
trabajadores_train = trabajadores.head(trabajadores.shape[0]-long_pred)
trabajadores_test = trabajadores.tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
trabajadores_train = TimeSeriesDataFrame(trabajadores_train)


# Definimos y ajustamos el modelo
modelo_2 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    ).fit(
    trabajadores_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_tiny", "ag_args": {"name_suffix": "tiny-ZeroShot"}},
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "small-ZeroShot"}}            
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_2.predict(trabajadores_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(trabajadores_test['target'], predictions['pred'])
score = interval_score(obs=trabajadores_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

trabajadores_test.drop('item_id', axis = 1, inplace = True)
trabajadores.columns = ['ds', 'y']
trabajadores_test.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
trabajadores_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : trabajadores_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_2.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_2_chronos = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}



# ------------------------------- 2.2 AJUSTE DEL MODELO CON HORIZONTE 3-------------------------------
# Largo del pronostico
long_pred = 3

# Divido en conjunto entrenamiento y prueba
trabajadores_train = trabajadores.head(trabajadores.shape[0]-12)
trabajadores_test = (trabajadores.head(trabajadores.shape[0]-12)).tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
trabajadores_train = TimeSeriesDataFrame(trabajadores_train)

# Definimos y ajustamos el modelo
modelo_2_3 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    ).fit(
    trabajadores_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_2_3.predict(trabajadores_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(trabajadores_test['target'], predictions['pred'])
score = interval_score(obs=trabajadores_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

trabajadores_test.drop('item_id', axis = 1, inplace = True)
trabajadores.columns = ['ds', 'y']
trabajadores_test.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
trabajadores_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : trabajadores_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_2_3.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_2_chronos3 = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}


# ------------------------------- 2.2 AJUSTE DEL MODELO CON HORIZONTE 6 -------------------------------
# Largo del pronostico
long_pred = 6

# Divido en conjunto entrenamiento y prueba
trabajadores_train = trabajadores.head(trabajadores.shape[0]-12)
trabajadores_test = (trabajadores.head(trabajadores.shape[0]-12)).tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
trabajadores_train = TimeSeriesDataFrame(trabajadores_train)

# Definimos y ajustamos el modelo
modelo_2_6 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    ).fit(
    trabajadores_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_2_6.predict(trabajadores_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(trabajadores_test['target'], predictions['pred'])
score = interval_score(obs=trabajadores_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

trabajadores_test.drop('item_id', axis = 1, inplace = True)
trabajadores.columns = ['ds', 'y']
trabajadores_test.columns = ['ds', 'y']
predictions.reset_index(drop=True, inplace=True)
trabajadores_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : trabajadores_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_2_6.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_2_chronos6 = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}

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
tiempo_rosario.columns = ['timestamp', 'target', 'HUM', 'PNM'] # % de Humedad y Presion a nivel del mar en hectopascales 

# Agrego una columna identificadora necesaria para Chronos
tiempo_rosario['item_id'] = 0

# Definicion del nivel de significacion y el largo del pronostico
alpha = 0.2

q_lower = alpha/2
q_upper = 1-alpha/2

# ------------------------------- 3.2 AJUSTE DEL MODELO -------------------------------
long_pred = 24

# Divido en conjunto entrenamiento y prueba
tiempo_rosario_train = tiempo_rosario.head(tiempo_rosario.shape[0]-long_pred)
tiempo_rosario_test = tiempo_rosario.tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
tiempo_rosario_train = TimeSeriesDataFrame(tiempo_rosario_train)

# Definimos y ajustamos el modelo
modelo_3 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    freq = "h"

    ).fit(
    tiempo_rosario_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_3.predict(tiempo_rosario_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(tiempo_rosario_test['target'], predictions['pred'])
score = interval_score(obs=tiempo_rosario_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

tiempo_rosario_test.drop('item_id', axis = 1, inplace = True)
tiempo_rosario.columns = ['ds', 'y', 'HUM', 'PNM']
tiempo_rosario_test.columns = ['ds', 'y', 'HUM', 'PNM']
predictions.reset_index(drop=True, inplace=True)
tiempo_rosario_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : tiempo_rosario_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_3.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_3_chronos = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}



# ------------------------------- 3.2 AJUSTE DEL MODELO CON HORIZONTE 6 -------------------------------
long_pred = 6

# Divido en conjunto entrenamiento y prueba
tiempo_rosario_train = tiempo_rosario.head(tiempo_rosario.shape[0]-24)
tiempo_rosario_test = (tiempo_rosario.head(tiempo_rosario.shape[0]-24)).tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
tiempo_rosario_train = TimeSeriesDataFrame(tiempo_rosario_train)

# Definimos y ajustamos el modelo
modelo_3_6 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    freq = "h"

    ).fit(
    tiempo_rosario_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_3_6.predict(tiempo_rosario_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(tiempo_rosario_test['target'], predictions['pred'])
score = interval_score(obs=tiempo_rosario_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

tiempo_rosario_test.drop('item_id', axis = 1, inplace = True)
tiempo_rosario.columns = ['ds', 'y', 'HUM', 'PNM']
tiempo_rosario_test.columns = ['ds', 'y', 'HUM', 'PNM']
predictions.reset_index(drop=True, inplace=True)
tiempo_rosario_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : tiempo_rosario_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_3_6.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_3_chronos6 = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}


# ------------------------------- 3.2 AJUSTE DEL MODELO CON HORIZONTE 12 -------------------------------
long_pred = 12

# Divido en conjunto entrenamiento y prueba
tiempo_rosario_train = tiempo_rosario.head(tiempo_rosario.shape[0]-24)
tiempo_rosario_test = (tiempo_rosario.head(tiempo_rosario.shape[0]-24)).tail(long_pred)

# Transformo el dataset a formato TimeSeriesDataFrame
tiempo_rosario_train = TimeSeriesDataFrame(tiempo_rosario_train)

# Definimos y ajustamos el modelo
modelo_3_12 = TimeSeriesPredictor(
    
    prediction_length=long_pred,
    quantile_levels =  [q_lower, q_upper],
    eval_metric = 'MAPE',
    freq = "h"

    ).fit(
    tiempo_rosario_train, 
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small"}
        ]
    },
    random_seed = seed,
    enable_ensemble = False
    )

# Realizamos las predicciones
predictions = modelo_3_12.predict(tiempo_rosario_train)
predictions.columns = ['pred', 'lower', 'upper']

# Calculamos MAPE e Interval Score
mape = mean_absolute_percentage_error(tiempo_rosario_test['target'], predictions['pred'])
score = interval_score(obs=tiempo_rosario_test['target'], lower=predictions['lower'], upper= predictions['upper'], alpha=0.2)


# Modificamos los datasets para tener el mismo formato que con el resto de modelos

tiempo_rosario_test.drop('item_id', axis = 1, inplace = True)
tiempo_rosario.columns = ['ds', 'y', 'HUM', 'PNM']
tiempo_rosario_test.columns = ['ds', 'y', 'HUM', 'PNM']
predictions.reset_index(drop=True, inplace=True)
tiempo_rosario_test.reset_index(drop=True, inplace=True)

pred_chronos = pd.DataFrame({
    'ds' : tiempo_rosario_test['ds'],
    'pred' : predictions['pred'],
    'lower' : predictions['lower'],
    'upper' : predictions['upper']
})

# Guardamos las metricas
salida = modelo_3_12.fit_summary()
grilla = salida['leaderboard']
tiempo = grilla['fit_time_marginal'][0]

# Guardamos los resultados
resultados_3_chronos12 = {'pred': pred_chronos, 'Horizonte': long_pred, 'mape': mape, 'score': score, 'tiempo': tiempo, 'grilla': grilla}


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# Guardamos los resultados
save_env(env_dict={
    'resultados_1_chronos' : resultados_1_chronos,
    'resultados_1_chronos3' : resultados_1_chronos3,
    'resultados_1_chronos6' : resultados_1_chronos6,
    'resultados_2_chronos' : resultados_2_chronos,
    'resultados_2_chronos3' : resultados_2_chronos3,
    'resultados_2_chronos6' : resultados_2_chronos6,
    'resultados_3_chronos' : resultados_3_chronos,
    'resultados_3_chronos6' : resultados_3_chronos6,
    'resultados_3_chronos12' : resultados_3_chronos12
    }, filename="Codigo/Ambiente/Amb_Aplicacion_chronos.pkl")

# Guardamos los modelos en un archivo a parte para no cargarlos innecesariamente
save_env(env_dict={
    'modelo_chronos_1' : modelo_1,
    'modelo_chronos_2' : modelo_2,
    'modelo_chronos_3' : modelo_3
    }, filename="Codigo/Ambiente/Modelos_chronos.pkl")
