from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

def predict_fun():
    spam_model = open('spam_model.pkl', 'rb')
    clf =