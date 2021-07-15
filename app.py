from flask import Flask, render_template, request
from preprocessingtest import PredictFeatures

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    doc = request.form['t']
    print(doc)
    p = PredictFeatures(doc)
    data = p.predict()

    return render_template('predict.html', data=data)

    

if __name__ == '__main__':
    app.run(debug=True)


