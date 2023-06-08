import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('cervical_cancer.csv')

# Split the dataset into input features (X) and target variable (y)
X = data.drop('risk_level', axis=1)
y = data['risk_level']

# Convert the input features to text using symptom descriptions
X_text = X.apply(lambda row: ' '.join(row.astype(str)), axis=1)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict risk level based on user input
def predict_risk_level(user_input):
    # Vectorize the user input
    user_input_vectorized = vectorizer.transform([user_input])

    # Predict the risk level
    predicted_risk = model.predict(user_input_vectorized)

    return predicted_risk[0]

# Get user input
print("Please describe your symptoms:")
user_input = input()

# Predict the risk level
predicted_risk = predict_risk_level(user_input)
print("Predicted risk level:", predicted_risk)
