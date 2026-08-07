import pandas as pd
import re
from razdel import tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

#loading data
df = pd.read_csv('ipo_riskfactors.csv')
print(df.head())
print(df['label'].value_counts())

#equalizing data
hit = df[df['label'] == 'HIT']
miss = df[df['label'] == 'MISS']

n = min(len(hit), len(miss))

hit = hit.sample(n=n, random_state=42)
miss = miss.sample(n=n, random_state=42)

df = pd.concat([hit, miss])

print(df['label'].value_counts())

#cleaning text
def clean_text(text):
    text = text.lower()                                 # lower register
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text)   # removing punctuation marks
    text = re.sub(r"\d+", " ", text)                    # removing digits
    text = re.sub(r"\s+", " ", text).strip()            # removing spaces
    return text

df['clean_text'] = df['risk_factors'].apply(clean_text)

#tokenizing
def tokenize_text(text):
    return [token.text for token in tokenize(text)]

df['tokens'] = df['clean_text'].apply(tokenize_text)


#removing stopwords
eng_stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [t for t in tokens if t not in eng_stopwords]

df['tokens_no_stop'] = df['tokens'].apply(remove_stopwords)

#lemmatizing

lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

df['lemmas'] = df['tokens_no_stop'].apply(lemmatize)
df['lemmas_joined'] = df['lemmas'].apply(lambda x: " ".join(x))

#stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def stemming(tokens):
    return [stemmer.stem(t) for t in tokens]
df['stems'] = df['tokens_no_stop'].apply(stemming)
df['stems_joined'] = df['stems'].apply(lambda x: " ".join(x))

#Vectorization

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['lemmas_joined'], df['label'], test_size=0.2, random_state=42)

bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model_bow = LogisticRegression(max_iter=1000)
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

print("=== Bag of Words ===")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\n=== TF-IDF ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))