# 😊 EmojiSense – Text to Emoji Prediction System

## 📌 Overview

**EmojiSense** is a Machine Learning project that predicts the most suitable emoji based on user input text.
It uses Natural Language Processing (NLP) techniques and a Neural Network model to understand text and map it to emojis.

---

## 🚀 Features

* 🔤 Text preprocessing using TF-IDF vectorization
* 🤖 Emoji prediction using Neural Network (MLPClassifier)
* 📊 Label encoding for emoji classification
* 💾 Saves trained model, encoder, and vectorizer
* 💬 Interactive CLI-based prediction

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib

---

## 📂 Project Structure

```
EmojiSense/
│
├── data/
│   └── emoji1.csv          # Dataset
│
├── emoji_model.pkl         # Trained model
├── emoji_encoder.pkl       # Label encoder
├── emoji_vectorizer.pkl    # TF-IDF vectorizer
│
├── emoji_sense.py          # Main script
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

* Contains text and corresponding emoji labels
* Example:

```
text,emoji
"I am happy", 😊
"I am sad", 😢
```

---

## ⚙️ How It Works

1. Load dataset using Pandas
2. Encode emoji labels using LabelEncoder
3. Convert text to numerical features using TF-IDF
4. Train Neural Network (MLPClassifier)
5. Save model and components using Joblib
6. Predict emoji for new user input

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/EmojiSense.git
cd EmojiSense
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Project

```
python emoji_sense.py
```

---

## 💡 Example

```
Enter text: I am feeling great today
Predicted Emoji: 😊
```

---

## 🔐 Important Note

* `.env` file is not included for security reasons
* Do not upload API keys or sensitive data

---

## 📈 Future Improvements

* 🌐 Web interface using Streamlit or Flask
* 🎯 Improve model accuracy with larger dataset
* 🧠 Use Deep Learning (LSTM / BERT)
* 🌍 Multi-language support

---

## 👩‍💻 Author

**Jerlin Ishabel**
MSc Data Science Student

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
