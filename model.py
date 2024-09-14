import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Example training data; replace with your actual training data
X_train = [
    "This is a spam email.",
    "This is a legitimate email.",
    # Add more samples here
]
y_train = [1, 0]  # 1 for spam, 0 for ham

# Initialize and train the model
vectorizer = TfidfVectorizer()
model = LogisticRegression()

# Create a pipeline with both vectorizer and model
pipeline = make_pipeline(vectorizer, model)
pipeline.fit(X_train, y_train)

# Save the model and vectorizer
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model and vectorizer have been trained and saved successfully.")
