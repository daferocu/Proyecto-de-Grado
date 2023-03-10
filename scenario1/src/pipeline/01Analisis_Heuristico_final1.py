#!/usr/bin/env python
# coding: utf-8

# # Análisis Heurístico

# Bajo el análisis eurístico se carga el trabajo realizado desde excel en donde se hace un filtrado supervisado de las variables del diccionario que se encuentran en "Nombre_Producto". A partir de ahí haremos un analisis univariante y bivariante.

# **Configuración de algoritmo:**
# - Lectura de datos en versión final, en formato .pickle.
# - Selección de grupos a partir de la variable nombre_producto.
# - Creación DataFrames.
# - Análisis Univariante.
# - Análisis Bivariante.
# - Anova: Análisis de la varianza.
# - Análisis de independencia: Tablas de contigencia.
# - Exportar base de datos segmentada.

# #### Importar librerías

# In[1]:


# Load Packages
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Importar la base de datos (.pickle)

# In[2]:


import pickle 
import random


# In[ ]:


# Suprimir alertas
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


print(sys.path)


# In[ ]:


sys.path.append("C:\\Users\\AnalistaNegB2B\\Desktop\\Analista Negocios B2B\\DVJ\\Maestria Ciencias de los Datos y Analitica\\2021-1\\PROYECTO INTEGRADOR\\proy-segmentacion")


# In[ ]:


from root import DIR_DATA
from root import DIR_CONF
from root import DIR_ROOT


# In[ ]:


input_fuente2 = DIR_DATA + "02-Raw/Fuente_2_vista_minable.pickle"


# In[7]:


df=pickle.load( open(input_fuente2, "rb" ) )


# In[8]:


df.shape


# In[9]:


df.nunique(axis=0)
print(df.nunique(axis=0))


# ### Selección de grupos a partir de la variable nombre_producto 
# A partir de la vista minable se construye una tabla dinámica que permita hacer un análisis eurístico en donde se hace un filtro de nombre_producto con las palabras seleccionadas en el diccionario y se analizan las familias, clases y productos que surgen de este filtro. 

# #### Diccionario:
# 
# Tecnologia, Informatica, Data, Datos, Analitica, Analizador, Dato, Prediccion

# #### Creamos DataFrame "df_familia"
# Incluye todas las familias que se encontraron con el filtro de productos. 

# In[10]:


df_familia = df[df.cod_familia.isin(['V1.32150000','V1.42000000','V1.43000000','V1.43200000','V1.43210000','V1.43220000',
'V1.43230000','V1.55120000','V1.56110000','V1.70100000','V1.70120000','V1.71000000','V1.71110000','V1.71150000','V1.72150000','V1.77100000', 'V1.80100000',
'V1.80110000','V1.80150000','V1.80160000','V1.81000000','V1.81100000','V1.81110000','V1.81160000','V1.82110000','V1.83000000','V1.83120000',
'V1.84140000','V1.93130000','V1.93140000','V1.93150000','V1.94100000'])]

df_familia.nunique(axis=0)
print(df_familia.nunique(axis=0))


# #### Creamos DataFrame "df_clase"
# Incluye todas las clase que se encontraron con el filtro de productos a partir de df_familia. 

# In[11]:


df_clase = df_familia[df_familia.cod_clase.isin(['V1.32152000','V1.42200000','V1.43210000','V1.43230000','V1.43201500','V1.43201800','V1.43211700',
'V1.43221700','V1.43223300','V1.43232300','V1.43232600','V1.43233400','V1.55121700','V1.56112000','V1.70101600','V1.70122000','V1.71150000','V1.71112300',
'V1.71151100','V1.72151600','V1.77101800','V1.80101500','V1.80111600','V1.80111700','V1.80151500','V1.80161500','V1.81110000','V1.81160000','V1.81101500',
'V1.81111500','V1.81111700','V1.81111800','V1.81111900','V1.81112000','V1.81112200','V1.81161800','V1.82111900','V1.83120000','V1.83121600','V1.84141700',
'V1.93131500','V1.93141800','V1.93151500','V1.94101800'])]


df_clase.nunique(axis=0)
print(df_clase.nunique(axis=0))


# #### Creamos DataFrame "df_producto"
# Incluye todas los productos que se encontraron con el filtro inicial a partir de df_clase. 

# In[12]:


df_producto = df_clase[df_clase.cod_producto.isin(['V1.32152002','V1.42201500','V1.42202500','V1.42203600','V1.43211700','V1.43212200',
'V1.43232300','V1.43233500','V1.43201509','V1.43201550','V1.43201814','V1.43211718','V1.43211730','V1.43211731','V1.43221721','V1.43223301',
'V1.43232304','V1.43232305','V1.43232307','V1.43232309','V1.43232311','V1.43232605','V1.43233402','V1.55121718','V1.56112001','V1.70101601',
'V1.70122010','V1.71151100','V1.71112303','V1.71151106','V1.72151605','V1.77101801','V1.80101507','V1.80111608','V1.80111609','V1.80111711',
'V1.80111712','V1.80111713','V1.80151503','V1.80161506','V1.81111700','V1.81111900','V1.81112000','V1.81161800','V1.81101512','V1.81111507',
'V1.81111704','V1.81111806','V1.81111902','V1.81112001','V1.81112002','V1.81112003','V1.81112006','V1.81112007','V1.81112009','V1.81112205',
'V1.81161801','V1.82111902','V1.83121600','V1.83121603','V1.83121604','V1.84141701','V1.93131503','V1.93141806','V1.93151502','V1.93151509',
'V1.94101803'])]


df_producto.nunique(axis=0)
print(df_producto.nunique(axis=0))


# In[13]:


print("Tamaño inicial:",df.shape, "Tamaño familia:",df_familia.shape, "Tamaño clases:",df_clase.shape, "Tamaño producto:", df_producto.shape)


# ## Análisis  Univariante

# In[14]:


df_producto.info()


# In[15]:


df_producto['precio_base'] = pd.to_numeric(df_producto['precio_base'])


# ### Describe

# In[16]:


df_producto.describe(include=['object'])


# ### Media, Mediana, Moda

# In[17]:


media = df_producto["precio_base"].mean()
mediana = df_producto["precio_base"].median()
moda = df_producto["precio_base"].mode()
print("""
    Media: %d
    Mediana: %d
    Moda: %d
""" % (media,mediana,moda))


# ### Desviación Estándar (SD)

# In[18]:


standard_deviation = df_producto['precio_base'].std()
print(standard_deviation)


# ## Análisis  Bivariante

# Ánalisis de boxplot entre Precio Base y Modalidad

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(10,8))
ax = sns.boxplot(x='modalidad', y='precio_base', data=df_producto, orient="v")
plt.savefig(DIR_ROOT + '/img/boxplotPrecioBase_Modalidad.png')


# Ánalisis de histograma en Precio Base

# In[20]:


filter_data = df_producto.dropna(subset=['precio_base'])
plt.figure(figsize=(14,8))
sns.distplot(filter_data['precio_base'], kde=False)
plt.savefig(DIR_ROOT + '/img/histogramaPrecioBase.png')


# Distribucion porcentual por Modalidad

# In[21]:


modalidades = df_producto['modalidad'].value_counts()
df2 = pd.DataFrame({'modalidades': modalidades}
                   )
df2.plot.pie(y='modalidades', figsize=(10,10), autopct='%1.1f%%')
plt.savefig(DIR_ROOT + '/img/DistribuporcentualModalidad.png')


# Histograma por Localización

# In[22]:


sns.set(style='darkgrid')
plt.figure(figsize=(20,10))
ax = sns.countplot(x='localizacion_desc', data=df_producto)
plt.savefig(DIR_ROOT + '/img/HistLocalización.png')


# ### Correlación de Pearson

# In[23]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(df_producto['ppto_global'], df_producto['precio_base'])
print("El coeficiente de correlación Pearson is", pearson_coef, " con un valor P de P =", p_value)  


# ### Anova: Análisis de la varianza
# Statsmodels - OLS
# 
# El nombre Anova: análisis de varianza se basa en el enfoque en el cual el procedimiento utiliza las varianzas para determinar si las medias son diferentes. El procedimiento funciona comparando la varianza entre las medias de los grupos y la varianza dentro de los grupos como una manera de determinar si los grupos son todos parte de una población más grande o poblaciones separadas con características diferentes.
# 
# La estadística F es simplemente un cociente de dos varianzas. Las varianzas son una medida de dispersión, es decir, qué tan dispersos están los datos con respecto a la media. Los valores más altos representan mayor dispersión.
# 
# Cuanto más dispersos estén los puntos, mayor será el valor de la variabilidad en el numerador del estadístico F.

# In[24]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# #### ANOVA Precio Base - Familia

# In[25]:


model = ols('precio_base ~ nombre_familia',                 # Model formula
            data = df_producto).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# #### ANOVA Precio Base - Localización

# In[26]:


model = ols('precio_base ~ localizacion',                 # Model formula
            data = df_producto).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# #### ANOVA Precio Base - Mes Proyectado

# In[27]:


model = ols('precio_base ~ mes_proyectado',                 # Model formula
            data = df_producto).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# #### ANOVA Precio Base - Modalidad

# In[28]:


model = ols('precio_base ~ modalidad',                 # Model formula
            data = df_producto).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
print (anova_result)


# ##### En conclusión:
# Las variables Familia, Mes Proyectado y Modalidad tienen un valor F alto, lo que las hace variables independientes no relacionadas en el modelo.

# ### Análisis de independencia: Tablas de contigencia
# 
# En estadística las tablas de contingencia se emplean para registrar y analizar la asociación entre dos o más variables, habitualmente de naturaleza cualitativa (nominales u ordinales).
# 
# Si la proporción de individuos en cada columna varía entre las diversas filas y viceversa, se dice que existe asociación entre las dos variables. Si no existe asociación se dice que ambas variables son independientes.

# #### Tabla de contingencia modalidad / producto

# In[29]:


pd.crosstab(index=df_producto['nombre_producto'],
            columns=df_producto['modalidad'], margins=True)


# #### Tabla de contingencia localizacion / producto

# In[30]:



pd.crosstab(index=df_producto['nombre_producto'],
            columns=df_producto['localizacion_desc'], margins=True)


# #### Tabla de contingencia Mes proyectado / producto

# In[31]:


pd.crosstab(index=df_producto['nombre_producto'],
            columns=df_producto['mes_proyectado'], margins=True)


# ##### Cálcular percentiles de Precio Base

# In[32]:


p25_precio = np.percentile(df_producto["precio_base"], 25)
p50_precio = np.percentile(df_producto["precio_base"], 50)
p75_precio = np.percentile(df_producto["precio_base"], 75)
p100_precio = np.percentile(df_producto["precio_base"], 100)
print("P25:",p25_precio,"P50:",p50_precio,"P75:",p75_precio,"P100:",p100_precio)


# #### Segmentar Precio Base por valores

# In[33]:


bins = [0,21000000,43827960,81804454,29550000000]

names = ["0-21.000.000","21.000.001-43.827.960", "43.827.961-81804454","81804454-2.9550.000.000"]


# In[34]:


df_producto['rango_precio'] = pd.cut(df_producto['precio_base'], bins, labels = names)


# In[35]:


df_producto['rango_precio'].fillna('0-21.000.000', inplace=True)


# In[36]:


sns.set(style='darkgrid')
plt.figure(figsize=(20,10))
ax = sns.countplot(x='rango_precio', data=df_producto)


# ## Exportar base de datos segmentada

# In[43]:


df_producto.to_pickle(DIR_DATA + '04-Refined/analisis_heuristico.pickle')


# In[ ]:


df_producto.to_csv(DIR_DATA + '04-Refined/analisis_heuristico.csv')

