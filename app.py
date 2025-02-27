import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import streamlit as st

# Load and prepare data
data = pd.read_csv(r'C:\Users\Hamza\Documents\Data Science\Natural language processing\Project\BBC News Classification\bbc_data.csv', on_bad_lines='skip')
data.drop_duplicates(inplace=True)
data['data'] = data['data'].str.lower()

# Define stopwords and stemmer globally
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    return re.sub(r'\(xc2xa3[\d\.]+m\)', '', text)

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

data['data'] = data['data'].apply(clean_text).apply(stemming)

# Label replacement
replacement_dict = {
    'sport': 0,
    'business': 1,
    'politics': 2,
    'entertainment': 3,
    'tech': 4
}
data['labels'] = data['labels'].replace(replacement_dict)

# Prepare data for training
x = data['data']
y = data['labels']

# Vectorization
tf = TfidfVectorizer(max_features=10000)  # Limit feature size
X = tf.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

from sklearn.tree import DecisionTreeClassifier
lr = DecisionTreeClassifier()
lr.fit(x_train, y_train)


# Streamlit app
st.set_page_config(page_title="BBC News Classifier")
st.header("BBC News Classifier📰📡")

input_text = st.text_input("Enter the BBC news you want to check: ")

if st.button("Classify"):
    if input_text:
        # Transform input text and make prediction
        input_data = tf.transform([input_text])
        prediction = lr.predict(input_data)[0]
        
        # Map prediction to category
        categories = {
            0: "Sports🏀",
            1: "Business📊",
            2: "Politics🏛️",
            3: "Entertainment🎬",
            4: "Tech🤖"
        }
        category = categories.get(prediction, "Unknown category")
        
        # Display result
        st.markdown(
            f"**News is related to: <span style='font-size:24px; color:#0cd0f2'>{category}</span>**",
            unsafe_allow_html=True)

    else:
        st.write("Please enter some text to classify.")
