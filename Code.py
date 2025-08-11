import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# View first few rows
print(df.head())

# Drop unnecessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

from sklearn.preprocessing import LabelEncoder

# Convert labels (ham/spam) to binary
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])  # ham=0, spam=1

# Check distribution
sns.countplot(data=df, x='label')
plt.title("Label Distribution")
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize messages
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label_num']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
