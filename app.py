import streamlit as st
import random
import string
import nltk
import numpy as np
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Caption Studio AI", page_icon="🚀", layout="centered")

st.title("🚀 AI Caption Generator")
st.write("Turn ideas into **viral captions** using NLP + BERT ✨")

# ---------------- LOAD ONLY ONCE (VERY IMPORTANT) ----------------
@st.cache_resource
def load_bert():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_bert()

# ---------------- NLP SETUP ----------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Words we NEVER want in captions (fixes your issue)
NOISE_WORDS = {
    "tokenization", "lemmatization", "stopword", "tagging",
    "processing", "nlp", "model", "embedding", "dataset"
}

# ---------------- EMBEDDING ----------------
def get_embedding(text):
    return np.mean(bert(text)[0], axis=0)

# ---------------- CLEAN NLP PIPELINE ----------------
def nlp_pipeline(text):

    # clean text
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    # remove stopwords
    filtered = [w for w in tokens if w not in stop_words]

    # lemmatize
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    sentence_vec = get_embedding(" ".join(lemmas))

    # limit words (speed + quality fix)
    unique_words = list(set(lemmas))[:12]

    scores = []
    for word in unique_words:
        vec = get_embedding(word)
        score = cosine_similarity([sentence_vec], [vec])[0][0]
        scores.append((word, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    # remove NLP-noise words
    keywords = []
    for w, _ in scores:
        if w not in NOISE_WORDS and len(w) > 2:
            keywords.append(w)

    # fallback
    if len(keywords) < 3:
        keywords = [w[0] for w in scores[:3]]

    return tokens, filtered, lemmas, keywords[:5]

# ---------------- VIRAL CAPTION ENGINE ----------------
def generate_caption(keywords):

    key = ", ".join(keywords)

    templates = [
        f"🚀 Just built something amazing using {key}",
        f"✨ Turning ideas into reality with {key}",
        f"💡 Built different with {key}",
        f"🔥 From concept to creation: {key}",
        f"🌿 Creating magic with {key}",
        f"⚡ AI-powered journey: {key}",
        f"🚀 Learning, building, shipping: {key}",
        f"💻 Modern AI build using {key}",
        f"🌟 Small idea → big impact with {key}",
        f"🔥 Code. Create. Innovate. ({key})"
    ]

    return random.choice(templates)

# ---------------- INPUT ----------------
text = st.text_area("💬 Enter your idea / project description")

# ---------------- BUTTON ----------------
if st.button("✨ Generate Viral Caption") and text:

    with st.spinner("Creating AI-powered caption... 🚀"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)
        caption = generate_caption(keywords)

    # ---------------- OUTPUT ----------------
    st.success(caption)

    # ---------------- DEBUG (CLEAN UI) ----------------
    with st.expander("🔑 Keywords"):
        st.write(keywords)

    with st.expander("🧠 Clean Words"):
        st.write(lemmas)

    with st.expander("🔤 Tokens"):
        st.write(tokens)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Built with NLP + BERT | Caption Studio AI 🚀")