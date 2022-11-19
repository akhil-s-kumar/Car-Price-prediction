from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    present_price = int(request.form['present_price'])
    fuel_type = int(request.form['fuel_type'])
    seller_type = int(request.form['seller_type'])
    year = int(request.form['year'])
    km_driven = int(request.form['km_driven'])
    prediction = model.predict([[present_price, fuel_type, seller_type, year, km_driven]])
    output = int(prediction[0])
    return render_template("index.html", prediction_text=f'Predicted price is {output}')

if __name__ == "__main__":
    app.run()