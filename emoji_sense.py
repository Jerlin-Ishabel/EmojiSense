import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\jjerl\Downloads\EmojiSense\data\emoji1.csv")  # Your dataset path
texts = df["text"].values
labels = df["emoji"].values

# -----------------------------
# 2. Encode labels
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=500)  # Can increase for bigger dataset
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# -----------------------------
# 5. Build and train MLP model
# -----------------------------
clf = MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# 6. Save model, encoder, vectorizer
# -----------------------------
joblib.dump(clf, "emoji_model.pkl")
joblib.dump(encoder, "emoji_encoder.pkl")
joblib.dump(vectorizer, "emoji_vectorizer.pkl")
print("✅ Model, encoder, and vectorizer saved!")

# -----------------------------
# 7. Predict new text
# -----------------------------
def predict_emoji(text):
    x = vectorizer.transform([text])
    pred = clf.predict(x)
    return encoder.inverse_transform(pred)[0]

# -----------------------------
# 8. Interactive Prediction
# -----------------------------
while True:
    user_text = input("\nEnter text (or 'quit' to stop): ")
    if user_text.lower() == "quit":
        break
    print("Predicted Emoji:", predict_emoji(user_text))
