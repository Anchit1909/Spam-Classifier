import json
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import time
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#Function to load lottie file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.image("logo.svg")
st.text("")
st.text("")

col1, col2, col3 = st.columns((1.5,.5,1))

with col1:
    st.title("Identify spam messages with a click of a button")
    st.markdown('Protect yourself from *getting spammed* by using this service.')
st.text("")
st.text("")

with col2:
    st.text("")
    st.text("")

with col3:
    lottie_spam = load_lottiefile("side_image.json")

    st_lottie(
        lottie_spam,
        speed=0.5,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        key=None,
        height=250,
        width=250,
    )

st.text("")
st.text("")
st.text("")

st.markdown("------------------------------------------")
st.text("")
st.text("")

col4, col5 = st.columns((1,1))
with col4:
    st.image("process_re.svg", width=300)
with col5:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.markdown("This application uses **Multinomial Naive Bayes** to classify messages into Spam or Not Spam.   "
                "*The **precision** of the result is 100% and the **accuracy** is 97.2%.*")

st.text("")
st.text("")
st.markdown("------------------------------------------")
st.text("")
st.text("")

st.header("Email/SMS Spam Classifier")
input_msg = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_msg = transform_text(input_msg)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_msg])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.error("This is a Spam message")
    else:
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.success("This is not a Spam Message")