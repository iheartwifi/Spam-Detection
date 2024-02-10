import pickle

# Créer un modèle fictif (remplacez cela par votre modèle réel)
model = {'paramètre': 42, 'autre_paramètre': 'abc'}

# Enregistrer le modèle dans un fichier
with open('Naive_move.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
