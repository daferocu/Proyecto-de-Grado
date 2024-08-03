# Load Packages
import os
import sys
import random
import pickle

## Data science libraries
import pandas as pd
import numpy as np
from datetime import datetime
from pandas import DataFrame, merge


#FIXME: Persona demo -> Validar que se esta importando archivo root.py
## Se usa para importar root.py para conectar carpeta data con carpeta src
# ROOT_PATH = os.path.dirname(
#     (os.sep).join(os.path.abspath(__file__).split(os.sep)[:-2]))
# sys.path.insert(1, ROOT_PATH)

#FIXME: Persona demo -> Validar que se esta importando archivo root.py
import root

# Tambien, se puede hacer de esta forma.
#from root import DIR_DATA
#from root import DIR_CONF

print("@@@ Imprimiendo ruta de DATA")
print(root.DIR_DATA)
print("@@@ Imprimiendo ruta de DATA_RAW")
print(root.DIR_DATA_RAW)
print("@@@ Imprimiendo ruta de DATA_STAGE")
#FIXME: Persona demo -> Hacer print de DIR_DAT_STAGE

print(root.DIR_DATA_STAGE)


#TODO Importacion inicial de archivo crudo desde data/stage/xx para luego generar pickle de vista minable
input_fuente2 = root.DIR_DATA_STAGE + "df_raw_fuente_2.pickle"

df=pickle.load( open(input_fuente2, "rb" ) )

# Renombrar las columnas v
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


# ## Creamos un indice de NIT
# Para asignar el nombre de la entidad matriz a cada registro
indice_nit = pd.read_csv(root.DIR_DATA_RAW + "indice_nit.csv", encoding= 'unicode_escape')

# #Renombrar variables
# indice_nit = indice_nit.rename(columns={'NIT': "nit",
#                                         'Entidad': "entidad_matriz"})

# # Se genera merge
# df = indice_nit.merge(df, on='nit', how='left')
# df

# # Eliminar filas de 'modalidad' que no le interesan a Caoba.
# df=df[df["modalidad"].str.contains("03|04|05|07|10|11|15|17|18")==False]
# print(df.modalidad.unique())

# # Reemplazar el 'mes_proyectado'==No Definido por 'Enero'
# df['mes_proyectado'].replace('No Definido','Enero', inplace=True)
# df['mes_proyectado'].value_counts()

# #Realizando casting a variables
# df[['date_last_publication','date_publised']]=df[['date_last_publication','date_publised']].astype('datetime64[ns]')
# df['date_last_publication'].head()

# # Ajustar el formato de 'date_publised' a año-mes-dia
# df['date_publised']=df['date_publised'].dt.strftime('%d/%m/%Y')
# df['date_publised']=pd.to_datetime(df['date_publised'],format='%d/%m/%Y')

# # **Consideremos los días entre los 2 campos 'date_last_publication' y 'date_publised':**
# df['diff_dates']=(df['date_last_publication']-df['date_publised']).dt.days
# #print(sorted(df.diff_dates.unique()))

# # Realizando cambios sobre variable year_publised
# df['year_publised'] = pd.DatetimeIndex(df['date_publised']).year

# df.dropna(inplace=True)


#TODO Exportando archivo serializado de Fuente_2_vista_minable.pickle. Debe usar extension .pickle o .pkl

#FIXME: Persona demo -> relacionar
#output_fuente2 = <> + "Fuente_2_vista_minable.pickle"
output_fuente2 = df_secopII.pickle + "Fuente_2_vista_minable.pickle"
#print(output_fuente2)
print(f'%%% Generando archivo serializado en data/stage/ de vista minable del dataframe => df')
print(f'%%% Archivo se encuentra en {output_fuente2}')
#FIXME: Persona demo -> Hacer print de DIR_DAT_STAGE

df.to_pickle(output_fuente2)

# # Fin de script n
