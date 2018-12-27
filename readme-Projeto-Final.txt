###############################
##### Software utilizados #####
###############################
IDE...: Anaconda 3.5
Editor.: Jupyter Notebook
Python: 3.7

###############################
#### Biliotecas utilizadas ####
###############################
# Numpy
import numpy as np
# Pandas usada para manipulação do dataframe
import pandas as pd
# Obter data e hora
from time import time
# Comando display
from IPython.display import display 
# Pré-Processamento, normalização de escala de valores
from sklearn.preprocessing import MinMaxScaler
# Biblioteca para gerar números randomicos
import random
# Modelos de machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Métricas para avaliação dos modelos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix 
# Splitar (divir) o dataset
from sklearn.model_selection  import train_test_split
# Geração de gráficos
import matplotlib.pyplot as plt
# Biblioteca para ajustar tamanho dos graficos/figuras
from matplotlib.pyplot import figure
# IQR
from scipy.stats import iqr
# Gerar arquivos csv
import csv
# Para o método PCA
from sklearn.decomposition import PCA
# Biblioteca para tratamento de warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline