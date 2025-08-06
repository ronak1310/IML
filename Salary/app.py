from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = pickle.load(open('salary_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Gender': request.form['gender'],
        'Education': request.form['education'],
        'Job_Title': request.form['jobtitle'],
        'Experience': float(request.form['experience']),
        'Age': int(request.form['age']),
        'Location': request.form['location']
    }

    input_df = pd.DataFrame([data])

    # Encoding categorical variables
    input_encoded = pd.get_dummies(input_df)

    # Reindex with training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    return jsonify({'salary': f'₹ {round(prediction, 2)}'})

if __name__ == '__main__':
    app.run(debug=True)
