# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Charger le modèle au démarrage de l'application
with open('Naive_model.pkl', 'rb') as model_file:
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
