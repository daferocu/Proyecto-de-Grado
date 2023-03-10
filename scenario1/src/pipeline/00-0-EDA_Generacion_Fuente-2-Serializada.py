## Python libraries
import os
import sys

## Data science libraries
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

## Use to import in google_colab
"""
# This cell will prompt you to connect this notebook with your google account.
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"

import os
base_dir = "/content/gdrive/My Drive/Colab Notebooks/CAOBA/Segmentation/"
print(base_dir)
"""


#TODO Importando y usando archivo root.py para autocontener solucion analitica y no generar problemas con manejo de data
## Se usa para importar root.py para conectar carpeta data con carpeta src
ROOT_PATH = os.path.dirname(
    (os.sep).join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

import root

print("@@@ Imprimiendo ruta de DATA")
print(root.DIR_DATA)
print("@@@ Imprimiendo ruta de DATA_RAW")
print(root.DIR_DATA_RAW)

#TODO Importacion inicial de archivo crudo desde data/raw/xx para luego generar .pickle

#Organizando tipos para variables cuando importadas.
dtypes_dict_secopII = {
     'Anno':                        np.int64,
     'Identificador PAA':           str,
     'Entidad':                     str,
     'NIT':                         np.int64,
     'Localización':                str,
     'DescripcionUbicacion':        str,
     'Mision/Vision':               str,
     'Perspectiva Estrategica':     str,
     'Presupuesto Menor Cuantia':   np.int64,
     'Presupuesto Minima Cuantia':  np.int64,
     'Presupuesto Global':          np.int64,
     'Fecha Primera Publicación':   str, #Posible date
     'Mes Proyectado':              str,
     'Identificador Item':          str,
     'Categoria Principal':         str,
     'Precio Base':                 np.int64,
     'Ultima Fecha Modificacion':   str, #Posible date
     'Version':                     str,
     'Referencia Contrato':        str,
     'Referencia Operacion':        str,
     'Fecha Publicacion':           str, #Posible date
     'Modalidad':                   str,
     'Contacto':                    str,
     'UNSPSC - Codigo Producto':    str,
     'UNSPSC - Nombre Producto':    str,
     'UNSPSC - Codigo Clase':       str,
     'UNSPSC - Nombre Clase':       str,
     'UNSPSC - Codigo Familia':     str,
     'UNSPSC - Nombre Familia':     str

}


#TODO Usando root.DIR_DATA_RAW luego de importar archivo root para leer data crudo
#Importando archivo crudo desde data/raw
df_secopII = pd.read_csv(root.DIR_DATA_RAW+"Plan_Anual_de_Adquisiciones_-_SECOP_II.csv",
                         dtype=dtypes_dict_secopII)
## Reanming columns
df_secopII  = df_secopII.rename(columns={
     'Anno':                        "year",
     'Identificador PAA':           "id_paa",
     'Entidad':                     "entidad",
     'NIT':                         "nit",
     'Localización':                "localizacion",
     'DescripcionUbicacion':        "localizacion_desc",
     'Mision/Vision':               "mision_vision",
     'Perspectiva Estrategica':     "pers_estrategica",
     'Presupuesto Menor Cuantia':   "pres_menor_cuantia",
     'Presupuesto Minima Cuantia':  "pres_min_cuantia",
     'Presupuesto Global':          "pres_global",
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
     'UNSPSC - Codigo Producto':    "cod_producto_unspsc",
     'UNSPSC - Nombre Producto':    "nombre_producto_unspsc",
     'UNSPSC - Codigo Clase':       "cod_clase_unspsc",
     'UNSPSC - Nombre Clase':       "nombre_clase_unspsc",
     'UNSPSC - Codigo Familia':     "cod_familia_unspsc",
     'UNSPSC - Nombre Familia':     "nombre_familia_unspsc"
})


## Punto 0: Entendiendo las variables discretas (categoricas), Better understanding of discrete variables

# To check unique values in variales
df_secopII .nunique(axis=0)
#print(df_secopII.nunique(axis=0))

df_secopII.describe()

# Variable ciiu
#print("Year")
#print(df_secopII.year.unique())

#print("\nId_paa")
# Variable macrosector
#print(df_secopII.id_paa.unique())

#print("\nEntidad")
# Variable macrosector
#print(df_secopII.entidad.unique())


#print("\nPrecio Base")
# Variable macrosector
#print(df_secopII.precio_base.unique())

#print("\nModalidad")
# Variable macrosector
#print(df_secopII.modalidad.unique())

#print("\nCategoria")
# Variable macrosector
#print(df_secopII.categoria_principal.unique())


#print("\nDate First Publication")
# Variable macrosector
#print(df_secopII.date_first_publication.unique())

#print("\nDate Last Publication")
# Variable macrosector
#print(df_secopII.date_last_publication.unique())

#print("\nDate Published")
# Variable macrosector
#print(df_secopII.date_publised.unique())

#print("\nMes Proyectado")
# Variable macrosector
#print(df_secopII.mes_proyectado.unique())

#TODO Exportando archivo serializado luego de leer crudo y hacerle EDA inicial
# Para exportar archivo serializado desde dataframe, se puede usar df.to_pickle()
print(f'%%% Generando archivo serializado en data/stage/ de dataframe => df_secopII')
print(f'%%% Archivo se encuentra en' +root.DIR_DATA_STAGE+"df_raw_fuente_2.pickle")
df_secopII.to_pickle(root.DIR_DATA_STAGE+"df_raw_fuente_2.pickle")


