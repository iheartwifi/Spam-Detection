#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

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

# Visualisation of the data
sns.countplot(data['spam'])
plt.show()
data.duplicated().sum()

# Separate X and Y
x = data['text'].values
y = data['spam'].values
print(y)

#Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

x_train.shape
y_train.shape

# Création du pipeline. 
# utilisé Pipeline pour créer un pipeline de traitement de données
# entraîné ce pipeline sur les données d'entraînement et j'ai prédit les données de test.

pipe = Pipeline([
    ('vect', CountVectorizer()),  # Utilisation de CountVectorizer pour convertir le texte en vecteurs
    ('clf', MultinomialNB()),  # Utilisation du classifieur Naive Bayes multinomial
])

#Train the pipeline
pipe.fit(x_train, y_train)

#Prediction on the test
y_pred = pipe.predict(x_test)

#Calculation of the precision
accuracy_score(y_pred, y_test)

#Exemple of prediction for an email
email = ["You won trip!"]
pipe.predict(email)

#Save the model
pickle.dump(pipe, open("Naive_model.pkl", 'wb'))
