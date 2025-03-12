from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# -------------------------
# Load data and train model
# -------------------------

# Read the CSV file
df = pd.read_csv('vgsales_5000.csv')

# Drop irrelevant columns
df = df.drop(columns=['Name', 'Year'])

# Define features and target
X = df.drop(columns=['Rank'])
y = df['Rank']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Specify the categorical and numerical features
categorical_features = ['Platform', 'Genre', 'Publisher']
numerical_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

# Preprocessing for categorical data with OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Create a pipeline with the preprocessor and RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Optional: Evaluate the model (prints to console)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# -------------------------
# Flask routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Retrieve form data ensuring field names match model features
        platform = request.form.get("platform")
        genre = request.form.get("genre")
        publisher = request.form.get("publisher")
        try:
            na_sales = float(request.form.get("na_sales"))
            eu_sales = float(request.form.get("eu_sales"))
            jp_sales = float(request.form.get("jp_sales"))
            other_sales = float(request.form.get("other_sales"))
            global_sales = float(request.form.get("global_sales"))
        except ValueError:
            return render_template("index.html", error="Please enter valid numbers for the sales fields.")
        
        # Prepare input DataFrame for prediction
        data = {
            "Platform": [platform],
            "Genre": [genre],
            "Publisher": [publisher],
            "NA_Sales": [na_sales],
            "EU_Sales": [eu_sales],
            "JP_Sales": [jp_sales],
            "Other_Sales": [other_sales],
            "Global_Sales": [global_sales]
        }
        input_df = pd.DataFrame(data)
        
        # Predict rank using the trained model
        predicted_rank = model.predict(input_df)
        prediction = round(predicted_rank[0], 2)
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
