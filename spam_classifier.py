import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    file_path = "cleaned_spam_data.csv"
    df = pd.read_csv(file_path, encoding='latin-1')
    return df

# Train model
def train_model(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Text'])
    y = df['Label']
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model and vectorizer
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    
    return model, vectorizer, accuracy

# Load and preprocess data
df = load_data()
model, vectorizer, accuracy = train_model(df)

# Streamlit UI
st.title("Spam Message Classifier")
st.write(f"Model Accuracy: {accuracy:.2f}")

# User Input
user_input = st.text_area("Enter message to classify:")

if st.button("Predict"):
    if user_input:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        result = "Spam" if prediction == 1 else "Ham"
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a message to classify.")
