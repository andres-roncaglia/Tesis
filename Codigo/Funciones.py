# ---------------------------------------- LIBRERIAS ------------------------------------

# Para el manejo de estructuras de datos
import pandas as pd
import numpy as np

# Para matematicas
from math import floor, ceil

# Para guardar y cargar los modelos
import pickle

# Para graficos
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_line, geom_ribbon, geom_histogram, geom_rect, geom_segment, geom_hline, geom_point, scale_x_continuous, scale_x_date, theme, theme_bw, element_blank, labs, scale_color_manual, scale_fill_manual, scale_y_continuous, element_text, theme_set, annotate, after_stat, facet_wrap


theme_set(theme_bw()+theme(plot_margin=0))

# Para autocorrelaciones
from statsmodels.tsa.stattools import acf, pacf

# Para calcular el test de ljung box
import statsmodels.api as sm

# Para la transformacion de box y cox y el test de normalidad
from scipy import stats

# Para crear tablas
from great_tables import GT, from_column, style, loc

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

def plot_forecast(data, forecast, pred_color = 'red', line_color = 'black', label = 'Prediccion', xlabel = 'Año', ylabel = 'Y', long=None, legend = True, legend_position  =(0.5,0.98)):

    # Guardamos el ultimo punto conocido para el pronostico para que no se vea tan abrupto el intervalo
    ult_punto = data.tail(len(forecast)+1).iloc[0]
    forecast = pd.concat([forecast, pd.Series({'ds': ult_punto['ds'], 'pred': ult_punto['y'], 'lower': ult_punto['y'], 'upper' : ult_punto['y']}).to_frame().T])

    # Nos quedamos con el largo deseado que mostrar
    if long != None:
        data_plt = data.copy()
        data_plt = data_plt.tail(long)
    else:
        data_plt = data
    
    # Unimos los datasets y nos aseguramos que esten en el formato correcto
    data_plt = pd.concat([data_plt, forecast])
    data_plt['ds'] = pd.to_datetime(data_plt['ds'])
    data_plt['y'] = pd.to_numeric(data_plt['y'])
    data_plt['pred'] = pd.to_numeric(data_plt['pred'])
    data_plt['lower'] = pd.to_numeric(data_plt['lower'])
    data_plt['upper'] = pd.to_numeric(data_plt['upper'])
    data_plt['fill'] = 'IC 80%'

    # Segun el nmbre del eje X voy a cambiar la frecuencia de este mismo
    if xlabel == 'Año':
      date_labels = "%Y"
      date_breaks = "1 year"
    elif xlabel == 'Día':
      date_labels = "%d"
      date_breaks = "1 day"

    # Cambio los breaks del eje x
    min_a = floor(data_plt[['y', 'lower', 'upper']].min().min())
    max_a = ceil(data_plt[['y', 'lower', 'upper']].max().max())
    breaks = [round(v) for v in np.linspace(min_a-(min_a%5), max_a+(5-max_a%5), 6)]
    
    return (
      ggplot(data_plt) +  
      
      geom_line(aes(x = 'ds', y = "y"), color = line_color, data=data_plt[data_plt['y'].notna()]) +
      
      geom_line(aes(x = 'ds', y = "pred"), color = pred_color, data=data_plt[data_plt['pred'].notna()]) +
      
      geom_ribbon(
        aes(ymin = 'lower', ymax = 'upper', x = 'ds'),
        fill = pred_color,
        alpha = 0.15,
        data=data_plt[data_plt['lower'].notna()]
        ) +
      
      geom_ribbon(
        aes(ymin = 'lower', ymax = 'upper', x = 'ds', fill = 'fill'),
        alpha = 0.15,
        data=data_plt[data_plt['lower'].notna()]
        ) +
      
      scale_fill_manual(breaks= ['IC 80%'], values = [pred_color]) +
      
      geom_line(aes(x = 'ds', y = "pred", color = 'fill'), data=data_plt[data_plt['pred'].notna()], alpha = 1) +
      
      scale_color_manual(breaks= ['IC 80%'], values = [pred_color]) +
      
      scale_x_date(date_labels = date_labels, date_breaks = date_breaks) +

      scale_y_continuous(breaks = breaks) +

      labs(x = xlabel, y = ylabel, fill=label, color = label) +
      theme(
        panel_grid_minor= element_blank(),
        legend_position = legend_position,
        axis_title=element_text(size = 8),
        axis_text= element_text(size = 7),
        legend_title=element_text(size = 8),
        legend_text= element_text(size = 6)
        )
    )

    
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
        ylab = r'$\rho_k$'
    else:
        autocorr = pacf(vector, nlags=lags, method = 'ywm')
        lags = np.arange(lags+1)

        autocorr = autocorr[1:]
        lags = lags[1:]
        ylab = r'$\phi_{kk}$'

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

    return(ggplot(autocorrelaciones) +
    aes(x = 'lag', y = atype) +
    geom_rect(ymin = lower_bound, ymax = upper_bound, xmax = np.inf, xmin = -np.inf, alpha = 0.01 ,fill = "#8ED081") +
    geom_rect(aes(ymin = -1, ymax = -np.inf, xmax = np.inf, xmin = -np.inf), alpha = 0.01 ,fill = "grey") +
    geom_rect(aes(ymin = 1, ymax = np.inf, xmax = np.inf, xmin = -np.inf), alpha = 0.01 ,fill = "grey") +
    geom_segment(aes(x = 'lag', xend = autocorrelaciones['lag'], yend = atype, y = 0)) +
    geom_hline(yintercept = 0) +
    geom_hline(color = "green",yintercept = upper_bound, linetype = "dashed") +
    geom_hline(color = "green",yintercept = lower_bound, linetype = "dashed") +
    geom_point(color = "#8ED081", size = 0.7) +
    geom_point(aes(x = 'lag', y = atype), color = col, size = 0.7) +
    scale_y_continuous(limits = (-1,1)) +
    scale_x_continuous(breaks = list(range(0,max(autocorrelaciones['lag'])+1, 2)), limits = (0,max(autocorrelaciones['lag']))) +
    labs(x = "Rezago (k)", y = ylab)
    )




# ------------------------------------------------------------------------------------
# Funcion resid_chek()
# Grafica la comprobacion de supuestos para los modelos arima

def resid_check(residuos_sin_estandarizar, ds, name, arima_df, time='%Y'):

    residuos_sin_estandarizar = residuos_sin_estandarizar.dropna()

    residuos = (residuos_sin_estandarizar - np.mean(residuos_sin_estandarizar))/ np.std(residuos_sin_estandarizar)
        
    # Histograma
    # Test de normalidad
    ks = stats.kstest(residuos,'norm').pvalue
    if ks < 0.0001:
        ks = '< 0.0001'
    else:
        ks = round(ks, 4)

    (ggplot() +
      aes(x = residuos) +
      geom_histogram(aes(y = after_stat('density')), color = "black", fill = "#B0D1E8", bins = 25) +
      labs(x = "Residuos estandarizados", y = "Densidad") +
      annotate(geom = "label", 
              label = f'Test de K-S\nP-value: {ks}',
              x = 2.5, y = 0.6, fill = "#D0E3F1", size = 6) +
      scale_x_continuous(limits = (-4,4)) +
      theme(
        axis_title=element_text(size = 7),
        axis_text= element_text(size = 5),
        legend_title=element_text(size = 6),
        legend_text= element_text(size = 4)
      )).save(f"../Imgs/plotnine/{name}_1.png", width=6/2.1, height=4/2.1, dpi=500)

    # Serie de los residuos
    
    outlayers = ((residuos > 3) | (residuos < -3))
    
    (ggplot() +
    aes(y = residuos, x = ds) +
    geom_line(color = "blue") +
    geom_hline(color = "black", linetype = "dashed", yintercept = 3) + 
    geom_hline(color = "black", linetype = "dashed", yintercept = -3) + 
    geom_point(color = "blue", size = 0.4) +
    geom_point(aes(x = ds[outlayers], y = residuos[outlayers]) , color = "red", size = 1.5) +
    scale_x_date(date_labels = time, date_breaks = "1 year" if time == '%Y' else '1 day') +
    labs(x = "Año", y = "Residuos estandarizados") +
      theme(
        axis_title=element_text(size = 7),
        axis_text= element_text(size = 5),
        legend_title=element_text(size = 6),
        legend_text= element_text(size = 4)
      )).save(f"../Imgs/plotnine/{name}_2.png", width=6/2.1, height=4/2.1, dpi=500)

    # Test de Ljung-box
    p_value = sm.stats.acorr_ljungbox(residuos, lags= 30, model_df=arima_df , return_df=True)['lb_pvalue'].min()
    if p_value < 0.0001:
        p_value = '< 0.0001'
    else:
        p_value = round(p_value, 4)

    # Autocorrelaciones
    lags = min(floor(residuos.dropna().shape[0]*0.49), 30)

    (autocorr_plot(residuos, lags=lags) +
      annotate(geom = "label", 
           label = f'Test de Ljung-Box\nMenor p-value: {p_value}',
           x = 15, y = -0.85, fill = "#D2EEDB", size = 6) +
      theme(
        axis_title=element_text(size = 7),
        axis_text= element_text(size = 5),
        legend_title=element_text(size = 6),
        legend_text= element_text(size = 4)
      )).save(f"../Imgs/plotnine/{name}_3.png", width=6/2.1, height=4/2.1, dpi=500)

    (autocorr_plot(residuos, lags=lags, atype='pacf') +
      theme(
        axis_title=element_text(size = 7),
        axis_text= element_text(size = 5),
        legend_title=element_text(size = 6),
        legend_text= element_text(size = 4)
      )).save(f"../Imgs/plotnine/{name}_4.png", width=6/2.1, height=4/2.1, dpi=500)




# ------------------------------------------------------------------------------------
# Funcion summary_to_latex()
# Hace que un pandas dataframe se convierta a formato latex para luego imprimirlo

def summary_to_latex(df, label = ''):
    fila_footnote = pd.DataFrame([{
        'Componente': r'\midrule Modelo',
        'Coeficiente': df['Modelo'].iloc[0],
        'IC(0.025)': '',
        'IC(0.975)': 'AIC',
        'p-value': df['AIC'].iloc[0]
    }])

    tabla_markdown = pd.concat(
        [df[['Componente','Coeficiente','IC(0.025)','IC(0.975)','p-value']], fila_footnote],
        ignore_index=True).to_latex(index=False, position="H", label=label)

    print(tabla_markdown)


# ------------------------------------------------------------------------------------
# Funcion summary_to_df()
# A partir de un modelo arima guarda la informacion del summary en un dataframe

def summary_to_df(model):
    df = pd.DataFrame({
      'Componente': model.params().index,
      'Coeficiente':round(model.params(), 4),
      'IC(0.025)':round(model.conf_int()[0], 4),
      'IC(0.975)':round(model.conf_int()[1], 4),
      'p-value':round(model.pvalues(), 4),
      'AIC':round(model.aic(), 4),
      'Modelo':f'SARIMA{model.get_params()['order']}{model.get_params()['seasonal_order']}',
      })
    
    return df


# ------------------------------------------------------------------------------------
# Funcion summary_to_html()
# Hace que un pandas dataframe se convierta a formato html, exclusivo para las presentaciones

def summary_to_html(df):
    fila_footnote = pd.DataFrame([{
        'Componente': 'Modelo',
        'Coeficiente': df['Modelo'].iloc[0],
        'IC(0.025)': '',
        'IC(0.975)': 'AIC',
        'p-value': df['AIC'].iloc[0]
    }])

    tabla_markdown = pd.concat(
        [df[['Componente','Coeficiente','IC(0.025)','IC(0.975)','p-value']], fila_footnote],
        ignore_index=True).to_html(index=False)

    print(tabla_markdown)

# ------------------------------------------------------------------------------------
# Funcion tabla_resumen()
# Resume las metricas de los pronosticos con distintos modelos

def tabla_resumen(metricas, path):

  metricas = metricas.drop(columns = 'Tiempo')

  t = metricas.pivot_table(
    index='Modelo',
    columns = 'Horizonte',
    values = ['MAPE', 'Interval Score']
  )
  t.columns = ['Interval_Score_3', 'Interval_Score_6', 'Interval_Score_12', 'MAPE_3', 'MAPE_6', 'MAPE_12']
  t = t[['Interval_Score_3', 'Interval_Score_6', 'Interval_Score_12', 'MAPE_3', 'MAPE_6', 'MAPE_12']]
  t['Modelo'] = t.index
  t = t.loc[metricas['Modelo'].drop_duplicates()]

  gt = (
      GT(t)
      .tab_stub(rowname_col="Modelo")
      .tab_stubhead(label="Modelo")
      .tab_spanner(
          label=f"Horizonte {int(metricas['Horizonte'].drop_duplicates()[0])}",
          columns=['MAPE_3', 'Interval_Score_3']
      )
      .tab_spanner(
          label=f"Horizonte {int(metricas['Horizonte'].drop_duplicates()[1])}",
          columns=['MAPE_6', 'Interval_Score_6']
      )
      .tab_spanner(
          label=f"Horizonte {int(metricas['Horizonte'].drop_duplicates()[2])}",
          columns=['MAPE_12', 'Interval_Score_12']
      )
      .fmt_number(decimals=4, sep_mark='')
      .cols_label(
        Interval_Score_3 = 'Interval Score',
        Interval_Score_6 = 'Interval Score',
        Interval_Score_12 = 'Interval Score',
        MAPE_3 = 'MAPE',
        MAPE_6 = 'MAPE',
        MAPE_12 = 'MAPE',
      )
      .opt_table_font(font=["Source Sans Pro", 'sans-serif'])
  )
  for col in t.columns:
      if col != "Modelo":
          fila = t[t[col] == t[col].min()].index.values[0]
          gt = gt.tab_style(
              style=style.fill(color="#B5E3C3"),
              locations=loc.body(columns=col, rows=[fila])
          )

  gt.save(path)


# ------------------------------------------------------------------------------------

# Funcion plot_forecast_compare()
# Grafica todos los pronosticos y los compara

def plot_forecast_compare(data, pronosticos, nombres_modelos, pred_color = 'red', line_color = 'black', xlabel = 'Año', ylabel = 'Y') :
    
  data_fin = pd.DataFrame(columns = ['ds', 'y', 'pred', 'lower', 'upper', 'fill', 'modelo'])

  for i in range(0,len(pronosticos)):

    # Unimos los datasets
    data_plt = pd.merge(data,pronosticos[i], on = 'ds', how='inner')
    data_plt['fill'] = 'IC 80%'
    data_plt['modelo'] = nombres_modelos[i]

    data_fin = pd.concat([data_fin, data_plt])

  # Corrijo los tipos de datos
  data_plt = data_fin
  data_plt['ds'] = pd.to_datetime(data_plt['ds'])
  data_plt['y'] = pd.to_numeric(data_plt['y'])
  data_plt['pred'] = pd.to_numeric(data_plt['pred'])
  data_plt['lower'] = pd.to_numeric(data_plt['lower'])
  data_plt['upper'] = pd.to_numeric(data_plt['upper'])

  # Agrego orden a los modelos
  cat_type = pd.CategoricalDtype(
    categories=nombres_modelos,
    ordered=True
  )
  data_plt['modelo'] = data_plt['modelo'].astype(cat_type)

  # Segun el nmbre del eje X voy a cambiar la frecuencia de este mismo
  if xlabel == 'Mes':
    date_labels = "%Y-%m"
    date_breaks = "3 month"
  elif xlabel == 'Hora':
    date_labels = "%m-%d, %HH"
    date_breaks = "6 hour"

  return(ggplot(data_plt) +  

    geom_line(aes(x = 'ds', y = "y"), color = line_color) +

    geom_line(aes(x = 'ds', y = "pred"), color = pred_color) +

    geom_ribbon(
      aes(ymin = 'lower', ymax = 'upper', x = 'ds'),
      fill = pred_color,
      alpha = 0.15,
      data=data_plt[data_plt['lower'].notna()]
      ) +

    geom_ribbon(
      aes(ymin = 'lower', ymax = 'upper', x = 'ds', fill = 'fill'),
      alpha = 0.15,
      data=data_plt[data_plt['lower'].notna()]
      ) +

    scale_fill_manual(breaks= ['IC 80%'], values = [pred_color]) +

    geom_line(aes(x = 'ds', y = "pred", color = 'fill'), alpha = 1) +

    scale_color_manual(breaks= ['IC 80%'], values = [pred_color]) +
    
    facet_wrap('modelo') +

    scale_x_date(date_labels = date_labels, date_breaks = date_breaks) +

    labs(x = xlabel, y = ylabel, fill='Pronóstico', color='Pronóstico') +

    # labs(x = xlabel, y = ylabel) +
    theme(
      panel_grid_minor= element_blank(),
      axis_title=element_text(size = 8),
      axis_text= element_text(size = 7),
      legend_title=element_text(size = 8),
      legend_text= element_text(size = 6),
      axis_text_x= element_text(angle = 60)
      )
  )