import streamlit as st
import random
import string
import nltk
import numpy as np
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Caption Studio AI",
    page_icon="🚀",
    layout="centered"
)

# ---------------- SAFE NLTK ----------------
def safe_downloads():
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet")

safe_downloads()

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #0f172a, #020617);
    color: #e5e7eb;
}

h1 {
    text-align: center;
    color: white !important;
}

textarea {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155;
}

.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# 🚀 Caption Studio AI")
st.markdown("Turn your ideas into **viral AI captions** using NLP + BERT ✨")

text = st.text_area("💬 Enter your project description")

generate = st.button("✨ Generate Viral Caption")

# ---------------- BERT MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_model()

# ---------------- HELPER: BERT EMBEDDING ----------------
def get_embedding(text):
    output = bert(text)
    return np.mean(output[0], axis=0)

# ---------------- NLP + BERT KEYWORD ENGINE ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    # ---------------- BERT sentence vector ----------------
    sentence_vec = get_embedding(" ".join(lemmas))

    # ---------------- semantic keyword scoring ----------------
    word_scores = []

    for word in set(lemmas):
        word_vec = get_embedding(word)
        score = cosine_similarity([sentence_vec], [word_vec])[0][0]
        word_scores.append((word, score))

    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

    keywords = [w[0] for w in word_scores[:6]]

    return tokens, filtered, lemmas, keywords

# ---------------- VIRAL CAPTION ENGINE ----------------
def generate_viral_caption(keywords):

    key_phrase = " ".join(keywords)

    captions = [
        f"🚀 Just built this using {key_phrase} 🔥",
        f"✨ From idea → reality with {key_phrase}",
        f"💡 Not just code… it's innovation powered by {key_phrase}",
        f"🔥 This project hits different: {key_phrase}",
        f"🌿 Built with passion using {key_phrase} 💻",
        f"⚡ AI in action: {key_phrase}",
        f"💻 Small idea → big impact using {key_phrase}",
        f"🌟 Turning concepts into reality with {key_phrase}",
        f"🔥 Built. Tested. Shipped. ({key_phrase})",
        f"🚀 Learning, building, repeating: {key_phrase}"
    ]

    return random.choice(captions)

# ---------------- MAIN APP ----------------
if generate and text:

    with st.spinner("Generating intelligent viral caption... 🚀✨"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)

        caption = generate_viral_caption(keywords)

    # ---------------- OUTPUT ----------------
    st.markdown(f"""
    <div class="card">
        <h3>💬 Your Viral Caption</h3>
        <h2>{caption}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- NLP INSIGHTS ----------------
    st.markdown("### 🌿 Behind the Magic")

    with st.expander("🔤 Tokens"):
        st.write(tokens)

    with st.expander("🚫 Filtered Words"):
        st.write(filtered)

    with st.expander("✍️ Lemmatized Words"):
        st.write(lemmas)

    with st.expander("🔑 BERT-Selected Keywords"):
        st.write(keywords)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio AI | NLP + BERT Semantic Engine | Viral Generator 🚀")