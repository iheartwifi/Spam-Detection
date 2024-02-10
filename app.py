#Dans cet exemple, app.py est le script principal de notre application web qui utilise Flask. 
#Nous avons une route principale '/' qui rend un modèle de formulaire où l'utilisateur peut saisir du texte. 
#Lorsque l'utilisateur soumet le formulaire, la route '/predict' est appelée, 
#où nous utilisons le modèle chargé pour effectuer une prédiction et renvoyer le résultat à l'utilisateur.


from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Charger le modèle au démarrage de l'application
with open('Naive_move.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
