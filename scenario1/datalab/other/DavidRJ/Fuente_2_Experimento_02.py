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

# In[1]:


# Load Packages
import os,sys
import pandas as pd
import numpy as np
from datetime import datetime


# In[2]:


import pickle 
import random


# In[3]:


# Mostrar la ruta del archivo indiferente del sistema operativo
import os
os.getcwd()


# In[4]:


main_path="C:\\Users\\David\\OneDrive\\EAFIT\\Proyecto Integrador 1\\Repositorios\\proy-segmentacion"
data_path="\\data\\03 - Trusted"


# In[5]:


os.chdir(main_path + data_path)


# In[6]:


df=pickle.load( open(r'analisis_heuristico.pickle', "rb" ) )
df.head()


# In[7]:


df.info


# In[8]:


df.dtypes


# In[9]:


# Validar los valores únicos de las variables
df.nunique(axis=0)
print(df.nunique(axis=0))


# ## Nueva variable rango_precio:

# In[10]:


df.shape


# In[11]:


# Variable rango_precio
print('\033[1m' + "rango_precio" + '\033[0m')
print(df.rango_precio.unique())


# En 'rango_precio' tenemos 4 categorías.

# ## K modes

# In[12]:


# supress warnings
import warnings
warnings.filterwarnings('ignore')

# Data viz lib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xticks


# In[13]:


df_cat=df[['entidad_matriz', 'localizacion_desc','mes_proyectado','modalidad','nombre_producto','nombre_clase','nombre_familia','rango_precio']]
df_cat.head()


# In[14]:


df_cat.shape


# In[15]:


df_cat.info()


# In[16]:


# Checking Null values
df_cat.isnull().sum()*100/df_cat.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# **Model Building**

# In[17]:


# First we will keep a copy of data
df_cat_copy = df_cat.copy()


# In[18]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_cat = df_cat.apply(le.fit_transform)
df_cat.head()


# In[19]:


get_ipython().system('pip install kmodes')


# In[20]:


# Importing Libraries
from kmodes.kmodes import KModes


# ### Using K-Mode with "Cao" initialization

# In[21]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(df_cat)


# In[22]:


# Predicted Clusters
fitClusters_cao


# In[23]:


clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = df_cat.columns


# In[24]:


# Mode of the clusters
clusterCentroidsDf


# ### Using K-Mode with "Huang" initialization

# In[25]:


km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(df_cat)


# In[26]:


# Predicted clusters
fitClusters_huang


# **Choosing K by comparing Cost against each K**

# In[27]:


cost = []
for num_clusters in list(range(1,10)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(df_cat)
    cost.append(kmode.cost_)


# In[28]:


y = np.array([i for i in range(1,10,1)])
plt.plot(y,cost)


# Al buscar un cambio de pendiente de empinada a poca profundidad (elbow), para determinar el número óptimo de clústeres, encontramos el primero en $K=2$, el segundo en $k=3$ y el tercero en $k=4$. Seleccionamos $k=3$. 

# ## Silhouette method

# The Silhouette coefficient of **+1** indicates that the sample is far away from the neighboring clusters.
# 
# The Silhouette coefficient of **0** indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
# 
# Silhouette coefficient **<0** indicates that those samples might have been assigned to the wrong cluster or are outliers.

# In[29]:


get_ipython().system(' pip install yellowbrick')


# In[30]:


# Silhouette Score for K modes
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer


# In[31]:


model = KModes()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,10),metric='silhouette', timings= True)
visualizer.fit(df_cat)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[32]:


df_cat.head()


# **Choosing K=2**

# In[33]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(df_cat)


# In[34]:


fitClusters_cao


# **Combining the predicted clusters with the original DF.**

# In[35]:


df_cat = df_cat_copy.reset_index()


# In[36]:


clustersdf = pd.DataFrame(fitClusters_cao)
clustersdf.columns = ['cluster_predicted']
combineddf = pd.concat([df_cat, clustersdf], axis = 1).reset_index()
combineddf = combineddf.drop(['index', 'level_0'], axis = 1)


# In[37]:


combineddf.head()


# ### Cluster Identification

# In[38]:


cluster_0 = combineddf[combineddf['cluster_predicted'] == 0]
cluster_1 = combineddf[combineddf['cluster_predicted'] == 1]
#cluster_2 = combineddf[combineddf['cluster_predicted'] == 2]


# In[39]:


cluster_0.info()


# In[40]:


cluster_1.info()


# ## Bar Plots

# **Localizacion descripcion**

# In[41]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['localizacion_desc'],order=combineddf['localizacion_desc'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.xticks(rotation='vertical')
plt.show()


# **Mes proyectado**

# In[42]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['mes_proyectado'],order=combineddf['mes_proyectado'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.show()


# **Modalidad**

# In[43]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['modalidad'],order=combineddf['modalidad'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.xticks(rotation='vertical')
plt.show()


# **Rango de Precio**

# In[44]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combineddf['rango_precio'],order=combineddf['rango_precio'].value_counts().index,hue=combineddf['cluster_predicted'])
plt.show()


# In[45]:


combineddf.head()


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


df_fin.to_csv("Cluster_final.csv")


# In[72]:


df_fin.to_pickle("Cluster_final.pickle")


# ## Fin experimento
