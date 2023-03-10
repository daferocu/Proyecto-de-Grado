# Loading libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from src.cmartinez.utils import getting_colnames_asrows
from root import DIR_DATA


def load_fuente1(input_f1):
	'''Obtencion de nombre de columnas como filas

    Args:
        df (pd.Dataframe): Dataframe para obtener las nombres de columns
	'''
	
	dtypes_dict_raw = {
		'No.':		np.int64,
		'NIT':    np.int64 , 
		'RAZON SOCIAL':		str, 
		'SUPERVISOR':		str, 
		'REGIÓN':		str, 
		'DEPARTAMENTO DOMICILIO':		str, 
		'CIUDAD DOMICILIO':		str, 
		'CIIU':		str, 
	  'MACROSECTOR':		str, 
	  'INGRESOS OPERACIONALES\n2018*':		np.float32, 
	  'GANANCIA (PERDIDA) 2018':		np.float32, 
	  'TOTAL ACTIVOS 2018':		np.float32,
	  'TOTAL PASIVOS 2018':		np.float32,
	  'TOTAL PATRIMONIO 2018':		np.float32,
	  'INGRESOS OPERACIONALES\n2017*':		np.float32, 
	  'GANANCIA (PERDIDA) 2017':		np.float32, 
	  'TOTAL ACTIVOS 2017':		np.float32, 
	  'TOTAL PASIVOS 2017':		np.float32,
	  'TOTAL PATRIMONIO 2017':		np.float32,
	  'GRUPO EN NIIF':		str
	}

	df_fuente1 = pd.read_csv(input_f1, dtype=dtypes_dict_raw)

	## Reanming columns for fuente1
	df_fuente1 = df_fuente1.rename(columns={
	    'No.':"ranking_given", 
	    'NIT':"nit", 
	    'RAZON SOCIAL':"razon_social", 
	    'SUPERVISOR':"supervisor", 
	    'REGIÓN':"region", 
	    'DEPARTAMENTO DOMICILIO':"departamento_dom", 
	    'CIUDAD DOMICILIO':"ciudad_dom", 
	    'CIIU':"ciiu", 
	    'MACROSECTOR':"macrosector", 
	    'INGRESOS OPERACIONALES\n2018*':"in_oper_2018", 
	    'GANANCIA (PERDIDA) 2018':"gap_2018", 
	    'TOTAL ACTIVOS 2018':"to_activo_2018", 
	    'TOTAL PASIVOS 2018':"to_pasivo_2018", 
	    'TOTAL PATRIMONIO 2018':"to_patri_2018", 
	    'INGRESOS OPERACIONALES\n2017*':"in_oper_2017", 
	    'GANANCIA (PERDIDA) 2017':"gap_2017", 
	    'TOTAL ACTIVOS 2017':"to_activo_2017", 
	    'TOTAL PASIVOS 2017':"to_pasivo_2017", 
	    'TOTAL PATRIMONIO 2017':"to_patri_2017", 
	    'GRUPO EN NIIF':"grupo_nif"
	})
	
	return(df_fuente1)

def procesar(df_fuente1,output_fuente1):
	# Ejecutando funcion de getting_colnames
	print("Variables en fuente 1")
	getting_colnames_asrows(df_fuente1)

	# Generando vista minable
	print("\nGenerando vista minable en path establecido")
	df_fuente1.to_pickle(output_fuente1)

# Ejecucion del proceso dado funciones anteriores
if __name__ == "__main__":

	input_fuente1 = DIR_DATA + "01-Trascient/1000_Empresas_mas_grandes_del_pa_s.csv"
	print("Cargando fuente1 en el path" + input_fuente1)
	df_fuente1 = load_fuente1(input_fuente1)
	output_fuente1 = DIR_DATA + "02-Raw/Fuente1_initial.pickle"
	print("Generando vista minable de fuente1")
	procesar(df_fuente1,output_fuente1)









