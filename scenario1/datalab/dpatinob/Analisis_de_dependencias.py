#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Load Packages
import os,sys
import pandas as pd
import numpy as np
from datetime import datetime

# This line is needed to display plots inline in Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Required for basic python plotting functionality
import matplotlib.pyplot as plt

# Required for formatting dates later in the case
import datetime
import matplotlib.dates as mdates

# Required to display image inline
from IPython.display import Image

# Advanced plotting functionality with seaborn
import seaborn as sns
sns.set(style="whitegrid") # can set style depending on how you'd like it to look


# In[23]:


df =pd.read_csv ("Fuente_2_vista_minable.csv")


# In[24]:


df.head(5)


# # Analisis de dependencia entre variables

# In[25]:


df.dtypes


# Las variables que veremos la si depende una de otra son las siguientes
# 
# 1 - entidad_matriz  (Esta tiene basicamente la misma informacion que entidad asi que solo se hara con esta)
# 
# 2 - localizacion_desc (Esta tiene la misma informacion que localizacion ya que del codigo viene la descripcion)
# 
# 3 - pers_estrategica (Esta tiene la misma informacion que mision_vision)
# 
# 4 - ref_contrato 
# 
# 5 - modalidad
# 
# 6 - contacto 
# 
# 7 - nombre_producto (tiene la misma info que cod_producto) 
# 
# 8 - nombre_clase (tiene la misma info que cod_clase)
# 
# 9 - nombre_familia (tiene la misma info que cod_familia)
# 

# In[31]:


Lista=["entidad_matriz","year","localizacion_desc","pers_estrategica","ref_contrato","modalidad","contacto","nombre_producto","nombre_clase","nombre_familia"]


# In[ ]:





# In[17]:


from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt


# In[32]:


for i in range(len(Lista)):
    for j in range(len(Lista)):
        
        if i != j:
            contigency= pd.crosstab(df[Lista[i]],df[Lista[j]])
            c, p, dof, expected = chi2_contingency(contigency)
            p=round(p,4)
            if p < 0.05:
                print(Lista[i],"Es dependiente a",Lista[j],"Bajo un nivel de confianza del 95% y un valor p de",p,"Estadistico",c)
            else:
                print(Lista[i],"Es independiente a",Lista[j],"Bajo un nivel de confianza del 95% y un valor p de",p,"Estadistico",c)                                   
        else:
            a=1


# In[27]:


import scipy.stats as stats


# # El resultado que nos dan es 

# Basicamente todas las variables son dependientes, quizimos probar manualmente con otro codigo para ver si era problema de la libreraria pero encontramos exactamente el mismo valor P y el mismo estadistico de prueba

# In[30]:


contigency= pd.crosstab(df[Lista[0]],df[Lista[1]])
c, p, dof, expected = chi2_contingency(contigency)
print(c,p)


# In[28]:


data_crosstab = pd.crosstab(df[Lista[0]],
                            df[Lista[1]],
                           margins=True, margins_name="Total")

# significance level
alpha = 0.05

# Calcualtion of Chisquare test statistics
chi_square = 0
rows = df[Lista[0]].unique()
columns = df[Lista[1]].unique()
for i in columns:
    for j in rows:
        O = data_crosstab[i][j]
        E = data_crosstab[i]['Total'] * data_crosstab['Total'][j] / data_crosstab['Total']['Total']
        chi_square += (O-E)**2/E

# The p-value approach
print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
p_value = 1 - stats.norm.cdf(chi_square, (len(rows)-1)*(len(columns)-1))
conclusion = "Failed to reject the null hypothesis."
if p_value <= alpha:
    conclusion = "Null Hypothesis is rejected."
        
print("chisquare-score is:", chi_square, " and p value is:", p_value)
print(conclusion)


# https://towardsdatascience.com/chi-square-test-with-python-d8ba98117626
# 
# https://medium.com/swlh/how-to-run-chi-square-test-in-python-4e9f5d10249d
