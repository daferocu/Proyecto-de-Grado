#!/usr/bin/env python
# coding: utf-8

# **Configuración de algoritmo:**
# - Lectura de datos en versión final, en formato .pickle.
# - Tokenizador, stopwords y Stemming
# - Distribución de frecuencia,Bigrams y trigrams para Mision-Visión ,Perspectiva Estrategica y nombre producto, respectivamente .
# - Lematización.
# - Modelo LDA con Producto (nombre_producto).
# - Modelo LDA con Mision-Visión (mision_vision).
# - Modelo LDA con Perspectiva Estrategica(pers_estrategica). 
# - Wordcloud
# - Exportar base de datos segmentada.

# In[4]:


#importar librerías
import os,sys
import pandas as pd
import numpy as np
import nltk
import re
import mcdm


# In[5]:


import pickle 
import random


# In[ ]:


# Suprimir alertas
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


print(sys.path)


# In[ ]:


sys.path.append("C:\\Users\\rojasd\\Desktop\\CAOBA\\Repositorios\\proy-segmentacion")


# In[ ]:


from root import DIR_DATA
from root import DIR_CONF
from root import DIR_ROOT


# In[ ]:


input_fuente2 = DIR_DATA + "03-Trusted/analisis_heuristico.pickle"


# In[ ]:


data2=pickle.load( open(input_fuente2, "rb" ) )


# In[10]:


# Renombrar las columnas 
data2  = data2.rename(columns={
     'year':                        "periodo", 
     
     
})


# In[12]:


data2.shape


# ### Tokenización

# Una base de datos de texto (o corpus) es una agrupación de bytes. El texto en su forma más pura, es una colección de bytes (o caracteres). La mayoria de veces es útil agrupar estos caracteres en unidades continuas llamadas tokens. En español, al igual que en la mayoria de los idiomas occidentales, un token corresponde a palabras y sequencias numericas separadas por espacios en blanco o signos de puntuación. El proceso de reducir un texto a tokens se conoce como tokenización.

# In[20]:


# settings en nltk: tokenizador y stopwords
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words_nltk = set(stopwords.words('spanish'))


# In[21]:


# función general para preparación de datos: tokenización, remoción caracteres especiales
# minusculas.... no incluido stemming y lematización.
import re

def textprep(data2):
    tokens = nltk.word_tokenize(str(data2))
    tokens = [w.lower() for w in tokens if len(w)>3]
    tokens = [re.sub(r'[^a-záéíóúñüÜ]+', '', w) for w in tokens]
    tokens = [re.sub(r'[í]+', 'i', w) for w in tokens]
    tokens = [re.sub(r'[á]+', 'a', w) for w in tokens]
    tokens = [re.sub(r'[é]+', 'e', w) for w in tokens]
    tokens = [re.sub(r'[ó]+', 'o', w) for w in tokens]
    tokens = [re.sub(r'[ú]+', 'u', w) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words_nltk] 
    return tokens


# #### Stemming

# In[22]:


#stemming

words  = nltk.tokenize.WhitespaceTokenizer().tokenize(str(data2))
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
#porter's stemmer
porterStemmedWords = [nltk.stem.PorterStemmer().stem(word) for word in words]
df['PorterStemmedWords'] = pd.Series(porterStemmedWords)
#SnowBall stemmer
snowballStemmedWords = [nltk.stem.SnowballStemmer("spanish").stem(word) for word in words]
df['SnowballStemmedWords'] = pd.Series(snowballStemmedWords)
df


# In[23]:


from nltk.tokenize import sent_tokenize

text=str(textprep(data2["mision_vision"]))
tokenized_sent= sent_tokenize(text)
print(tokenized_sent)


# In[24]:


tokens = [t for t in text.split()]

freq = nltk.FreqDist(tokens)


for key,val in freq.items():

    print (str(key) + ':' + str(val))


# In[25]:


freq.plot(20, cumulative=False,title='Distribución de frecuencia para los 20 tokens más comunes en Mision-Visión')


# ###### Bigrams Collocation Mision-Visión
# 
# Es posible que deseemos ver qué términos se usan juntos a menudo. Podemos hacer esto buscando colocaciones en Mision-Visión, es decir, dos símbolos de palabras que aparecen juntas en el texto con más frecuencia de lo que cabría esperar por casualidad.

# In[26]:


from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens, 20)


# In[27]:


finder.apply_freq_filter(10)


# In[28]:


finder.nbest(bigram_measures.likelihood_ratio, 10)


# #### Trigrams Perpectiva Mision-Visión
# 
# 
# Combinaciones frecuentes de tres palabras

# In[29]:


finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)


# In[30]:


finder.ngram_fd.most_common(5)


# In[31]:


finder.ngram_fd.tabulate(5)


# ##### Perspectiva Estrategica

# In[33]:


from nltk.tokenize import sent_tokenize

text=str(textprep(data2["pers_estrategica"]))
tokenized_sent= sent_tokenize(text)
print(tokenized_sent)


# In[34]:


tokens = [t for t in text.split()]

freq = nltk.FreqDist(tokens)


for key,val in freq.items():

    print (str(key) + ':' + str(val))


# In[35]:


freq.plot(20, cumulative=False,title='Distribución de frecuencia para los 20 tokens más comunes en Perpectiva Estrategica')


# ###### Bigrams Collocation Perpectiva Estrategica
# 
# Es posible que deseemos ver qué términos se usan juntos a menudo. Podemos hacer esto buscando colocaciones en Perpectiva Estrategica, es decir, dos símbolos de palabras que aparecen juntas en el texto con más frecuencia de lo que cabría esperar por casualidad.

# In[36]:


bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens, 20)


# In[37]:


finder.apply_freq_filter(10)


# In[38]:


finder.nbest(bigram_measures.likelihood_ratio, 10)


# #### Trigrams Perpectiva Estrategica
# 
# Combinaciones frecuentes de tres palabras

# In[39]:


finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens        )


# In[40]:


finder.ngram_fd.most_common(5)


# In[41]:


finder.ngram_fd.tabulate(5)


# ##### NombreProducto

# In[45]:


from nltk.tokenize import sent_tokenize

text=str(textprep(data2["nombre_producto"]))
tokenized_sent= sent_tokenize(text)
print(tokenized_sent)


# In[46]:


tokens = [t for t in text.split()]

freq = nltk.FreqDist(tokens)


for key,val in freq.items():

    print (str(key) + ':' + str(val))


# In[47]:


freq.plot(20, cumulative=False, title='Distribución de frecuencia para los 20 tokens más comunes en Nombre Producto')


# ###### Bigrams Collocation Producto
# 
# Es posible que deseemos ver qué términos se usan juntos a menudo. Podemos hacer esto buscando colocaciones en Nombre_producto, es decir, dos símbolos de palabras que aparecen juntas en el texto con más frecuencia de lo que cabría esperar por casualidad.
# 

# In[48]:


bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens, 20)


# In[49]:


finder.apply_freq_filter(10)


# In[50]:


finder.nbest(bigram_measures.likelihood_ratio, 10)


# #### Trigrams producto
# 
# Combinaciones frecuentes de tres palabras

# In[51]:


finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)


# In[52]:


finder.ngram_fd.most_common(5)


# In[53]:


finder.ngram_fd.tabulate(5)


# #### Lematización

# Lematización es el proceso de creación de lemas. Los Lemas corresponde a la raíz de una palabra. Considere por ejemplo la palabra correr. Esta puede tener distintas formas como corriendo, corrí, correré, entre otras. En este caso, resulta útil reducir la palabra a su raíz o lemma.

# In[57]:


nltk.download('wordnet')


# In[58]:


from nltk.stem.wordnet import WordNetLemmatizer
lem= WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
# escriba la palabra que necesite
word = "misiones"
print("lemmatized word:", lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


# In[59]:


words  = nltk.tokenize.WhitespaceTokenizer().tokenize(str(data2))
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
#WordNet Lemmatization
wordNetLemmatizedWords = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
df['WordNetLemmatizer'] = pd.Series(wordNetLemmatizedWords)
df


# ### LDA 
# 
# 
# 
# En esta sección se hará el análisis LDA (Latent Dirichlet Allocation) para la detección de tópicos, y dentro de los tópicos que arroje el modelo seleccionar los que se asocian con transformación digital. .

# #### Modelo LDA con Producto (nombre_producto)

# In[59]:


# creación de columna con tokenización de una columna de interés especifica
data2['tokens_nombre_producto'] = data2.apply(lambda row: textprep(row['nombre_producto']), axis=1)
data2.head()


# In[60]:


# Creación del BoW - en gensim es Dictionary
from gensim.corpora import Dictionary
dictionary = Dictionary(data2.tokens_nombre_producto)


# In[61]:


# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in data2.tokens_nombre_producto]


# In[65]:


print(dictionary)


# In[66]:


# libreria para paralelizar
import multiprocessing as mp
import time

t0 = time.time()
pool = mp.Pool(mp.cpu_count())
doc_term_matrix = pool.map(dictionary.doc2bow, [sentence for sentence in data2.tokens_nombre_producto])
pool.close()
print(time.time()-t0)


# In[68]:


from gensim.models.ldamulticore import LdaMulticore

t0 = time.time()
lda_model = LdaMulticore(doc_term_matrix, num_topics=20, id2word = dictionary, passes=10, workers=10)
print(time.time()-t0)


# In[69]:


def assigntopic(doc):
    vector = lda_model[dictionary.doc2bow(doc)] 
    # opción 1: todos los tópicos ordenados de mayor a menor, podria ser topN tambien asi: return vector[:5] n=5
    vector = sorted(vector, key=lambda item: -item[1])
    # opción 2: asignar el tópico mayor a cada documento
    #vector = max(vector,key=lambda item: item[1])
    return vector


# In[70]:


data2['topics'] = data2.apply(lambda row: assigntopic(row['tokens_nombre_producto']), axis=1)
data2.head()


# In[71]:


# Mostrar los términos y sus pesos de un documento
print(list(lda_model[doc_term_matrix[0]]))


# In[88]:


print (lda_model)


# In[89]:


# Mostrar los términos más relevantes de los tópicos más relevantes tópico y sus pesos
print(lda_model.print_topics(num_topics=10, num_words=5))


# In[90]:


lda_topic_assignment = [max(p,key=lambda item: item[1]) for p in lda_model[corpus]]


# In[91]:


import pyLDAvis.gensim_models as gensim
import pyLDAvis

t0 = time.time()
pyLDAvis.enable_notebook()
vis = gensim.prepare(lda_model, doc_term_matrix, dictionary, sort_topics = False)
print(time.time()-t0)
vis
vis.savefig(DIR_ROOT + '/img/da_visualizatioproductofinal.png')


# In[ ]:


pyLDAvis.save_html(vis, 'lda_visualizatioproductofinal.html')


# #### Modelo LDA con Perspectiva estrategica

# In[96]:


# creación de columna con tokenización de una columna de interés especifica
data2['tokens_pers_estrategica'] = data2.apply(lambda row: textprep(row['pers_estrategica']), axis=1)
data2.head()


# ### Construir el BoW (diccionario) de términos

# In[97]:


# Creación del BoW - en gensim es Dictionary
from gensim.corpora import Dictionary
dictionary = Dictionary(data2.tokens_pers_estrategica)


# In[98]:


# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in data2.tokens_pers_estrategica]


# In[99]:


print(dictionary)


# 
# ### Construir matriz de documentos vs términos

# In[100]:


# libreria para paralelizar
import multiprocessing as mp
import time

t0 = time.time()
pool = mp.Pool(mp.cpu_count())
doc_term_matrix = pool.map(dictionary.doc2bow, [sentence for sentence in data2.tokens_pers_estrategica])
pool.close()
print(time.time()-t0)


# ### Construir modelo LDA

# In[102]:


from gensim.models.ldamulticore import LdaMulticore

t0 = time.time()
lda_model = LdaMulticore(doc_term_matrix, num_topics=20, id2word = dictionary, passes=10, workers=10)
print(time.time()-t0)


# In[103]:


def assigntopic(doc):
    vector = lda_model[dictionary.doc2bow(doc)] 
    # opción 1: todos los tópicos ordenados de mayor a menor, podria ser topN tambien asi: return vector[:5] n=5
    vector = sorted(vector, key=lambda item: -item[1])
    # opción 2: asignar el tópico mayor a cada documento
    #vector = max(vector,key=lambda item: item[1])
    return vector


# In[105]:


data2['topics2'] = data2.apply(lambda row: assigntopic(row['tokens_pers_estrategica']), axis=1)
#data2.head()


# Ejemplos de tópicos del modelo

# In[106]:


# Mostrar los términos y sus pesos de un documento
print(list(lda_model[doc_term_matrix[0]]))


# In[107]:


print (lda_model)


# In[108]:


# Mostrar los términos más relevantes de los tópicos más relevantes tópico y sus pesos
print(lda_model.print_topics(num_topics=10, num_words=5))


# In[109]:


lda_topic_assignment = [max(p,key=lambda item: item[1]) for p in lda_model[corpus]]


# ### Visualización de todos los tópicos

# In[110]:


import pyLDAvis.gensim_models as gensim
import pyLDAvis

t0 = time.time()
pyLDAvis.enable_notebook()
vis = gensim.prepare(lda_model, doc_term_matrix, dictionary, sort_topics = False)
print(time.time()-t0)
vis
vis.savefig(DIR_ROOT + '/img/lda_visualizationestrategicafinal.png')


# ### Guardar la visualización en un archivo HTML 

# In[ ]:


pyLDAvis.save_html(vis, 'lda_visualizationestrategicafinal.html')


# #### Modelo LDA con Misión_visión

# In[115]:


# creación de columna con tokenización de una columna de interés especifica # 3 minutos y 30 segundos Cargando
data2['tokens_mision_vision'] = data2.apply(lambda row: textprep(row['mision_vision']), axis=1)
#data2.head()


# In[116]:


# Creación del BoW - en gensim es Dictionary
from gensim.corpora import Dictionary
dictionary = Dictionary(data2.tokens_mision_vision)


# In[117]:


# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in data2.tokens_mision_vision]


# In[118]:


# libreria para paralelizar
import multiprocessing as mp
import time

t0 = time.time()
pool = mp.Pool(mp.cpu_count())
doc_term_matrix = pool.map(dictionary.doc2bow, [sentence for sentence in data2.tokens_mision_vision])
pool.close()
print(time.time()-t0)


# In[ ]:


from gensim.models.ldamulticore import LdaMulticore 
#3 minutos cargando
t0 = time.time()
lda_model = LdaMulticore(doc_term_matrix, num_topics=20, id2word = dictionary, passes=10, workers=10)
print(time.time()-t0)


# In[ ]:


def assigntopic(doc):
    vector = lda_model[dictionary.doc2bow(doc)] 
    # opción 1: todos los tópicos ordenados de mayor a menor, podria ser topN tambien asi: return vector[:5] n=5
    vector = sorted(vector, key=lambda item: -item[1])
    # opción 2: asignar el tópico mayor a cada documento
    #vector = max(vector,key=lambda item: item[1])
    return vector


# In[ ]:


data2['topics3'] = data2.apply(lambda row: assigntopic(row['tokens_mision_vision']), axis=1)
#data2.head()
#1 minuto y medio cargado


# In[ ]:


# Mostrar los términos y sus pesos de un documento
print(list(lda_model[doc_term_matrix[0]]))


# In[ ]:


print (lda_model)


# In[ ]:


# Mostrar los términos más relevantes de los tópicos más relevantes tópico y sus pesos
print(lda_model.print_topics(num_topics=10, num_words=5))


# In[ ]:


lda_topic_assignment = [max(p,key=lambda item: item[1]) for p in lda_model[corpus]]


# In[ ]:


import pyLDAvis.gensim_models as gensim
import pyLDAvis

t0 = time.time()
pyLDAvis.enable_notebook()
vis = gensim.prepare(lda_model, doc_term_matrix, dictionary, sort_topics = False)
print(time.time()-t0)
vis
vis.savefig(DIR_ROOT + '/img/lda_visualizationmisiovisionfinal.png')


# In[ ]:


pyLDAvis.save_html(vis, 'lda_visualizationmisiovisionfinal.html')


# In[96]:


# Make WordCloud

text = ""
for word in data2['nombre_producto']:
    text += f'{word} '
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig(DIR_ROOT + '/img/WordCloudLDA.png')


# Exportar base de datos segmentada

# In[ ]:


data2.to_csv(DIR_DATA + '04-Refined/LDA_final01.csv')


# In[ ]:


data2.to_pickle(DIR_DATA + '04-Refined/LDA_final01.pickle')

