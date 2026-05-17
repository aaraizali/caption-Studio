import streamlit as st
import nltk
import random
import string
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- SAFE DOWNLOADS ----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------- UI ----------------
st.set_page_config(page_title="Caption Generator", page_icon="🚀", layout="centered")

st.title("🚀 AI Caption Generator")
st.write("Turn your ideas into **viral captions using NLP + BERT** ✨")

# ---------------- LOAD BERT (NOT USED HEAVILY) ----------------
@st.cache_resource
def load_bert():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_bert()

# ---------------- NLP PIPELINE ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    # simple keyword logic (BACK TO ORIGINAL STYLE)
    keywords = lemmas[:5]

    return tokens, filtered, lemmas, keywords

# ---------------- CAPTION ENGINE ----------------
def generate_caption(keywords):

    key = " ".join(keywords)

    templates = [
        f"🚀 Just built something amazing with {key}",
        f"✨ Turning ideas into reality using {key}",
        f"💡 Built with passion: {key}",
        f"🔥 New AI project completed: {key}",
        f"🌿 From idea to execution using {key}",
        f"⚡ Learning & building with {key}",
        f"🚀 Small idea → big impact: {key}",
        f"💻 Crafted using modern AI tools: {key}",
        f"🌟 Code. Create. Repeat. ({key})"
    ]

    return random.choice(templates)

# ---------------- INPUT ----------------
text = st.text_area("💬 Enter your project description")

if st.button("✨ Generate Caption") and text:

    with st.spinner("Generating caption... 🚀"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)
        caption = generate_caption(keywords)

    # ---------------- OUTPUT ----------------
    st.success(caption)

    with st.expander("🔍 Keywords"):
        st.write(keywords)

    with st.expander("🧠 Clean Words"):
        st.write(lemmas)

    with st.expander("🔤 Tokens"):
        st.write(tokens)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Built with NLP + BERT | Clean Caption Generator 🚀")