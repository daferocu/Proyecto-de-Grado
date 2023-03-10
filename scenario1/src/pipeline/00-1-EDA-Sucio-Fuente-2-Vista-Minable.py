#!/usr/bin/env python
# coding: utf-8

# # EDA - Fuente de datos 2
# ## Plan Anual de Adquisiones - Secop II
# ## Vista Minable

# **Configuración de algoritmo:**
# - Lectura de datos, en formato .pickle.
# - Creación de un indice de NIT.
# - Eliminar variables y registros no necesarios.
# - Depurar por modalidad de contratación.
# - Graficos de variables temporales.

# **Importar librerías.**

# In[1]:


# Load Packages
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pandas import DataFrame, merge


# This line is needed to display plots inline in Jupyter Notebook
#get_ipython().run_line_magic('matplotlib', 'inline')

# Required for basic python plotting functionality
import matplotlib.pyplot as plt

# Required for formatting dates later in the case
import datetime
import matplotlib.dates as mdates

# Advanced plotting functionality with seaborn
import seaborn as sns

# **Importar la base de datos (.pickle).**

import pickle
import random

# Organizando root
ROOT_PATH = os.path.dirname(
    (os.sep).join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

# Importing root
from root import DIR_DATA
from root import DIR_CONF

input_fuente2 = DIR_DATA + "02-Raw/df_raw_fuente_2.pickle"

# In[8]:
df=pickle.load( open(input_fuente2, "rb" ) )

# ## Organizar los datos para una vista minable.

# In[12]:


# Renombrar las columnas
df  = df.rename(columns={
     'Anno':                        "year",
     'Identificador PAA':           "id_paa",
     'Entidad':                     "entidad",
     'NIT':                         "nit",
     'Localización':                "localizacion",
     'DescripcionUbicacion':        "localizacion_desc",
     'Mision/Vision':               "mision_vision",
     'Perspectiva Estrategica':     "pers_estrategica",
     'Presupuesto Menor Cuantia':   "ppto_menor_cuantia",
     'Presupuesto Minima Cuantia':  "ppto_min_cuantia",
     'Presupuesto Global':          "ppto_global",
     'Fecha Primera Publicación':   "date_first_publication",
     'Mes Proyectado':              "mes_proyectado",
     'Identificador Item':          "id_item",
     'Categoria Principal':         "categoria_principal",
     'Precio Base':                 "precio_base",
     'Ultima Fecha Modificacion':   "date_last_publication",
     'Version':                     "version",
     'Referencia Contrato':         "ref_contrato",
     'Referencia Operacion':        "ref_operacion",
     'Fecha Publicacion':           "date_publised",
     'Modalidad':                   "modalidad",
     'Contacto':                    "contacto",
     'UNSPSC - Codigo Producto':    "cod_producto",
     'UNSPSC - Nombre Producto':    "nombre_producto",
     'UNSPSC - Codigo Clase':       "cod_clase",
     'UNSPSC - Nombre Clase':       "nombre_clase",
     'UNSPSC - Codigo Familia':     "cod_familia",
     'UNSPSC - Nombre Familia':     "nombre_familia"
})


# In[13]:


print(list(df.columns))


# In[14]:


# Validar los valores únicos de las variables
df.nunique(axis=0)
print(df.nunique(axis=0))


# In[15]:


df.info()


# ## Creamos un indice de NIT
# Para asignar el nombre de la entidad matriz a cada registro

# In[16]:


indice_nit = pd.read_csv(DIR_DATA + "02-Raw/indice_nit.csv", encoding= 'unicode_escape')
indice_nit.head()


# Aseguramos que la variable "NIT" conserve el mismo formato en ambos lados y modificamos el nombre de entidad por "entidad_matriz"

# In[17]:


#Renombrar variables
indice_nit = indice_nit.rename(columns={'NIT': "nit", 'Entidad': "entidad_matriz"})
indice_nit


# Agregamos un merge entre la matriz de entidades y el dataframe.

# In[18]:




# In[19]:


df = indice_nit.merge(df, on='nit', how='left')
df


# ## Eliminar variables y registros no necesarios

# In[20]:


#Eliminar variables
df=df.drop(['id_paa', 'ppto_menor_cuantia', 'ppto_min_cuantia', 'date_first_publication', 'id_item', 'categoria_principal', 'version','ref_operacion'], axis=1)
df.head(10)


# ## Depurar por modalidades

# In[21]:


print('\033[1m' + "modalidad" + '\033[0m')
print(df.modalidad.unique())


# In[22]:


# Eliminar filas de 'modalidad' que no le interesan a Caoba.
df=df[df["modalidad"].str.contains("03|04|05|07|10|11|15|17|18")==False]
df.head()


# In[23]:


print(df.modalidad.unique())


# In[24]:


df.shape


# In[25]:


# Reemplazar el 'mes_proyectado'==No Definido por 'Enero'
df['mes_proyectado'].replace('No Definido','Enero', inplace=True)
df['mes_proyectado'].value_counts()


# Convertimos las 2 variables de fecha, de tipo texto a tipo fecha:

# In[26]:


df[['date_last_publication','date_publised']].dtypes


# In[27]:


df[['date_last_publication','date_publised']]=df[['date_last_publication','date_publised']].astype('datetime64[ns]')
df['date_last_publication'].head()


# In[28]:


df['date_publised'].head()


# In[29]:


# Ajustar el formato de 'date_publised' a año-mes-dia
df['date_publised']=df['date_publised'].dt.strftime('%d/%m/%Y')
df['date_publised']=pd.to_datetime(df['date_publised'],format='%d/%m/%Y')
df['date_publised'].head()


# **Consideremos los días entre los 2 campos 'date_last_publication' y 'date_publised':**

# In[30]:


df['diff_dates']=(df['date_last_publication']-df['date_publised']).dt.days
df['diff_dates'].head(10)


# In[31]:


print(sorted(df.diff_dates.unique()))


# In[32]:


# Descripción de la nueva variable 'diff_dates'
df['diff_dates'].describe()


# In[33]:

df['diff_dates'].plot(kind='hist', bins=20)
plt.title('Diferencia en días entre fecha de publicación y última publicación')
plt.ylabel('Frecuencia')
plt.xlabel('Número de días')
plt.show()


# In[34]:


df[['year','diff_dates']].plot(kind='scatter',x='year',y='diff_dates')
plt.xticks(np.linspace(2016,2020,5,endpoint=True))
plt.show()


# **Comparación entre año de presupuesto (PAA) y año de publicación**

# In[35]:


df['year_publised'] = pd.DatetimeIndex(df['date_publised']).year
df['year_publised'].head()


# In[36]:


plt.scatter(df['year_publised'],df['year'])
plt.title('Dispersión entre año de PAA y de publicación')
plt.ylabel('Año de PAA')
plt.xlabel('Año de publicación')
plt.xticks(np.linspace(2016,2020,5,endpoint=True))
plt.yticks(np.linspace(2016,2020,5,endpoint=True))
plt.show()


# In[37]:


plt.scatter(df['year'],df['year_publised'])
plt.title('Dispersión entre año de PAA y de publicación')
plt.xlabel('Año de PAA')
plt.ylabel('Año de publicación')
plt.xticks(np.linspace(2016,2020,5,endpoint=True))
plt.yticks(np.linspace(2016,2020,5,endpoint=True))
plt.show()


# **Agrupemos algunas variables por fechas:**

# In[38]:


df.groupby(["year", 'year_publised'])["modalidad"].count()


# In[39]:


# Groupby "modalidad"; "localizacion"; "nombre_producto"


# ## Eliminar registros con valores NaN en el DataFrame

# In[40]:


df.head(10)


# In[41]:


df.dropna(inplace=True)
df

# In[42]:
output_fuente2 = DIR_DATA + "02-Raw/Fuente_2_vista_minable.pickle"
print(output_fuente2)
df.to_pickle(output_fuente2)

# # Fin de documento
