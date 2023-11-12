import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/home/', methods=("GET", "POST"))
def home():
    output = []
    data = []
    if request.method == 'POST':
        data.append(float(request.form["crim"]))
        data.append(float(request.form["zn"]))
        data.append(float(request.form["indus"]))
        data.append(float(request.form["chas"]))
        data.append(float(request.form["nox"]))
        data.append(float(request.form["rm"]))
        data.append(float(request.form["age"]))
        data.append(float(request.form["dis"]))
        data.append(float(request.form["rad"]))
        data.append(float(request.form["tax"]))
        data.append(float(request.form["ptratio"]))
        data.append(float(request.form["b"]))
        data.append(float(request.form["lstat"]))
        final_input = scalar.transform(np.array(data).reshape(1,-1))
        output = model.predict(final_input)
        #return redirect(url_for('home'))
    return render_template('home.html', prediction_text="The house price is {}".format(output))




    #print("TEST")
    #if request.method == 'POST':
    # data = [float(x) for x in request.form.values()]
    # final_input = scalar.transform(np.array(data).reshape(1,-1))
    # print(final_input)
    # output = model.predict(final_input)[0]
        

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    return jsonify(output[0])



if __name__ == "__main__":
    app.run(debug=True)

    
