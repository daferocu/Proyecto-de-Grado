#!/usr/bin/env python
# coding: utf-8

# # Fuente de datos 2
# ## Plan Anual de Adquisiones - Secop II¶

# **Configuración de algoritmo:**
# - Lectura de datos en versión final, en formato .pickle.
# - Descripción de tipo de datos.
# - Cambio en nombres de variables.
# - Identificación de valores únicos por variable.
# - Creación de agrupación (clusters) con K-modes.
# - Validación del número de clusters (k) óptimo con método Elbow y Silhouette.
# - Combinar con los datos originales.
# - Gráficos de barras para los clusters creados. 
# - Exportar base de datos segmentada.


# Importar paquetes
import os,sys
import pandas as pd
import numpy as np
from datetime import datetime

import pickle 
import random

# Suprimir alertas
import warnings
warnings.filterwarnings('ignore')

# Librerias de vizualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from matplotlib.pyplot import xticks

# Importar librerias
from kmodes.kmodes import KModes
from sklearn import preprocessing

# Score Silueta para K modes
from yellowbrick.cluster import KElbowVisualizer

print(sys.path)

sys.path.append("C:\\Users\\David\\OneDrive\\EAFIT\\Proyecto Integrador 1\\Repositorios\\proy-segmentacion")

from root import DIR_DATA
from root import DIR_CONF
from root import DIR_ROOT

input_fuente2 = DIR_DATA + "03-Trusted/analisis_heuristico.pickle"

df=pickle.load( open(input_fuente2, "rb" ) )

# Validar los valores únicos de las variables
df.nunique(axis=0)
print(df.nunique(axis=0))


# ## Nueva variable rango_precio:

# In[14]:


df.shape


# In[15]:


# Variable rango_precio
print('\033[1m' + "rango_precio" + '\033[0m')
print(df.rango_precio.unique())


# En 'rango_precio' tenemos 4 categorías.

# ## K modes

# In[16]:


df_cat=df[['entidad_matriz', 'localizacion_desc','mes_proyectado','modalidad','nombre_producto','nombre_clase','nombre_familia','rango_precio']]


# In[17]:


df_cat.shape


# In[18]:


# Validar valores nulos
df_cat.isnull().sum()*100/df_cat.shape[0]


# **Construcción del Modelo**

# In[19]:


# Hacemos una copia de los datos
df_cat_copy = df_cat.copy()


# In[20]:


le = preprocessing.LabelEncoder()
df_cat = df_cat.apply(le.fit_transform)
#df_cat.head()


# ### Usar K-Mode con inicialización "Cao"

# In[21]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(df_cat)


# In[22]:


# Predicción de Clusters
fitClusters_cao


# In[23]:


clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = df_cat.columns


# In[24]:


# Modos de los clústers
clusterCentroidsDf


# ### Usar K-Mode con inicialización "Huang"

# In[25]:


km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(df_cat)


# In[26]:


# Predicción de Clusters
fitClusters_huang


# **Seleccionar K comparando Costo contra cada K**

# In[27]:


cost = []
for num_clusters in list(range(1,10)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(df_cat)
    cost.append(kmode.cost_)


# In[36]:


y = np.array([i for i in range(1,10,1)])
plt.plot(y,cost)
plt.savefig(DIR_ROOT + '/img/codo01.png')


# In[34]:



print(DIR_ROOT)


# Al buscar un cambio de pendiente de empinada a poca profundidad (elbow), para determinar el número óptimo de clústeres, encontramos el primero en $K=2$, el segundo en $k=3$ y el tercero en $k=4$. Seleccionamos $k=3$. 

# ## Método Silueta

# Un coeficiente de Silueta **+1** indica que la muestra está muy lejos de los clusters vecinos.
# 
# Un coeficiente de Silueta de **0** indica que la muestra está muy cerca o sobre el límite de decisión entre dos clusters vecinos.
# 
# Un coeficiente de Silueta **<0** indica que la muestra pueden estar asignadas al cluster equivocado o que son outliers.

# In[31]:


model = KModes()
# k es el rango de números de clústers.
visualizer = KElbowVisualizer(model, k=(2,10),metric='silhouette', timings= True)
visualizer.fit(df_cat)        # Ajustar los datos al vizualizador
visualizer.show()        # Finalizar la figura
plt.savefig(DIR_ROOT + '/img/silueta01.png')


# **Seleccionamos K=2**

# In[33]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(df_cat)


# In[34]:


fitClusters_cao


# **Combinmos el clúster predecido con el DF original.**

# In[35]:


df_cat = df_cat_copy.reset_index()


# In[36]:


clustersdf = pd.DataFrame(fitClusters_cao)
clustersdf.columns = ['cluster_predicted']
combineddf = pd.concat([df_cat, clustersdf], axis = 1).reset_index()
combineddf = combineddf.drop(['index', 'level_0'], axis = 1)


# In[37]:


#combineddf.head()


# ### Identificación de Cluster 

# In[38]:


cluster_0 = combineddf[combineddf['cluster_predicted'] == 0]
cluster_1 = combineddf[combineddf['cluster_predicted'] == 1]


# In[39]:


cluster_0.info()


# In[40]:


cluster_1.info()


# ## Gráficos de Barras

# **Descripción de localización**

# In[41]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['localizacion_desc'],order=combineddf['localizacion_desc'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.xticks(rotation='vertical')
plt.show()
plt.savefig(DIR_ROOT + '/img/Localizacion01.png')


# **Mes proyectado**

# In[42]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['mes_proyectado'],order=combineddf['mes_proyectado'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.show()
plt.savefig(DIR_ROOT + '/img/Mes_proyectado01.png')


# **Modalidad**

# In[43]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['modalidad'],order=combineddf['modalidad'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.xticks(rotation='vertical')
plt.show()
plt.savefig(DIR_ROOT + '/img/Modalidad01.png')


# **Rango de Precio**

# In[44]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['rango_precio'],order=combineddf['rango_precio'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.show()
plt.savefig(DIR_ROOT + '/img/Rango_precio01.png')


# In[45]:


#combineddf.head()


# ## Vista de clusters final

# In[65]:


df_fin = df.reset_index()


# In[66]:


df_fin = df_fin.drop(['index'], axis = 1)


# In[67]:


cluster_predicted=combineddf["cluster_predicted"]
df_fin = df_fin.join(cluster_predicted)


# In[69]:


df_fin.head()


# In[70]:


df_fin.shape


# ## Exportar base de datos segmentada

# In[71]:


df_fin.to_csv(DIR_DATA + '04-Refined/Cluster_final01.csv')


# In[72]:


df_fin.to_pickle(DIR_DATA + '04-Refined/Cluster_final01.pickle')


# ## Fin cluster
