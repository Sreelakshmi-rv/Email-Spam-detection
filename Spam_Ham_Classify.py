import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

file_path = "D:/Downloads/spam (1).csv"
data = pd.read_csv(file_path, encoding='latin-1')
data.head()

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

pickle.dump(model, open("spam_model.pkl", "wb"))
print("✅ Model training complete and saved as 'spam_model.pkl'")