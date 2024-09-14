import pickle

# Load the vectorizer from the saved file
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Example email content to transform
email_content = "Special offer just for you"

# Transform the new email content using the loaded vectorizer
email_features = vectorizer.transform([email_content])

print("Email content has been transformed successfully.")
