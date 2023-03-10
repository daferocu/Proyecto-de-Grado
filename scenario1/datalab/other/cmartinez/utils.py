"""
    :Date: 29-06-202
    :Version: 0.1
    :Author: Camila Martinez <ex-mcmartinez@javeriana.edu.co>
    :Organization: Centro de Excelencia y Apropiación
    de Big Data y Data Analytics - CAOBA
"""


def getting_colnames_asrows(df):
    """Obtencion de nombre de columnas como filas

    Args:
        df (pd.Dataframe): Dataframe para obtener las nombres de columns
    """
    for colname in list(df.columns):
        print(colname)
