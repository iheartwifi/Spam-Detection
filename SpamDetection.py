#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#CSV file path
file = 'C:\\Users\\Admin\\OneDrive\\Pictures\\Documents\\GitHub\\Spam Detection\\emails.csv'

#open and read CSV file 
data = pd.read_csv(file)
data.head()

# Afficher la forme (nombre de lignes et de colonnes) du DataFrame
print(data.shape)

# Afficher le contenu de la colonne 'texte' de la première ligne
print(data['text'][0])

# Afficher le décompte des valeurs de la colonne 'spam'
print(data['spam'].value_counts())

sns.countplot(data['spam'])

data.duplicated().sum()

# Separate X and Y
x = data['text'].values
y = data['spam'].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

x_train.shape
y_train.shape

print(x_train.shape, y_train.shape)


#preprocessing
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

#train by ML algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
nb = MultinomialNB

pipe = make_pipeline(cv, nb)

pipe.fit(x_train, y_train)
