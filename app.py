import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the model and scaler once at the start
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('churn_scaler.pkl', 'rb'))

# Load the dataset to get the feature names
df = pd.read_csv('Cleaned_Telco-Customer-Churn.csv')
feature_names = df.drop('Churn Value', axis=1).columns.tolist()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    print("Features received:", features)

    new_df = pd.DataFrame([features], columns=feature_names)
    print("New DataFrame:", new_df)

    numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    for col in numerical_cols:
        try:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        except ValueError:
            return render_template('home.html', error=f"Invalid input for {col}")
    new_df = new_df.dropna()

    # One hot encode the categorical columns:
    categorical_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Multiple Lines', 'Internet Service',
                        'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
                        'Contract', 'Paperless Billing', 'Payment Method']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(new_df[categorical_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=new_df.index)
    print("Encoded Categorical DataFrame:", cat_encoded)
        
    num_scaled = pd.DataFrame(scaler.transform(new_df[numerical_cols]), columns=numerical_cols, index=new_df.index)
    print("Scaled Numerical DataFrame:", num_scaled)

    df_concat = pd.concat([cat_df, num_scaled], axis=1)
    print("Concatenated DataFrame:", df_concat)

    final_df = df_concat.reindex(columns=model.feature_names_in_)
    print("Final DataFrame for Prediction:", final_df)

    prediction = model.predict(final_df)
    probability = model.predict_proba(final_df)[:, 1]
    print("Prediction:", prediction)
    prediction = prediction.item()
    print("prediction type:", type(prediction))
    print("Probability:", probability)
    probability = probability.item()
    print("probability type:", type(probability))

    if prediction == 1:
        o1 = "This customer is likely to be churned!"
        o2 = f"Confidence: {probability * 100:.2f}%"
    else:
        o1 = "This customer is likely to continue!"
        o2 = f"Confidence: {probability * 100:.2f}%"

    return render_template('results.html', output1=o1, output2=o2)

if __name__ == "__main__":
    app.run(debug=True)