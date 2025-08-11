# 📧 Email Spam Detection with Machine Learning

The **Email Spam Detection** project uses supervised machine learning to classify emails as **Spam** or **Not Spam** based on their content. This type of system is essential in email filtering services to reduce unwanted or malicious emails.

## 📂 Dataset
The dataset consists of labeled email messages along with their text content. Preprocessing steps include:
- Removing punctuation and stopwords
- Converting text to lowercase
- Applying TF-IDF vectorization for feature extraction

## ⚙️ Implementation
The project follows these main steps:
1. **Data Loading** – Reading the spam dataset (`spam.csv`)
2. **Data Cleaning** – Removing unnecessary characters and formatting
3. **Feature Extraction** – Using TF-IDF to convert text into numerical form
4. **Model Training** – Applying machine learning algorithms like:
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
5. **Evaluation** – Checking accuracy, precision, recall, and F1-score

## 🚀 Usage
1. Clone the repository:
   ```bash
   git clone (https://github.com/arpit-02qc/Email-Spam-Detection-with-Machine-Learning/edit/main/README.md)
