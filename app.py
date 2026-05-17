import streamlit as st
import random
import string
import nltk
import numpy as np
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Caption Studio",
    page_icon="✨",
    layout="centered"
)

# ---------------- SAFE NLTK DOWNLOAD ----------------
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

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- UI STYLE (UNCHANGED) ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #0f172a, #020617);
    color: #e5e7eb;
    font-family: Arial;
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
    padding: 10px;
    font-weight: bold;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 14px;
    margin-top: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER (UNCHANGED) ----------------
st.markdown("# ✨Caption Studio")
st.markdown("Turn your ideas into **viral AI captions** ✨")

text = st.text_area("💬 Enter your project description here:")

generate = st.button("✨ Generate Viral Caption")

# ---------------- BERT MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_model()

# ---------------- EMBEDDING HELPER ----------------
def get_embedding(text):
    return np.mean(bert(text)[0], axis=0)

# ---------------- NLP PIPELINE (FIXED + SMART) ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    filtered = [w for w in tokens if w not in stop_words]

    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    clean_words = [w for w in lemmas if len(w) > 2]

    # limit for performance
    clean_words = clean_words[:20]

    # sentence meaning embedding
    sentence_embedding = get_embedding(" ".join(clean_words))

    # score words by semantic similarity
    scores = []
    for w in clean_words:
        try:
            word_embedding = get_embedding(w)
            score = np.dot(sentence_embedding, word_embedding)
            scores.append((w, score))
        except:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)

    keywords = [w[0] for w in scores[:6]]

    return tokens, filtered, lemmas, keywords

# ---------------- VIRAL CAPTION ENGINE (MEANING BASED) ----------------
def generate_viral_caption(keywords):

    key_phrase = " ".join(keywords)

    captions = [
        f"🧠 Understanding insights from {key_phrase}",
        f"📊 Exploring patterns in {key_phrase}",
        f"🚀 AI system analyzing {key_phrase}",
        f"💡 A step toward understanding {key_phrase}",
        f"📚 Research-driven insights on {key_phrase}",
        f"🔍 Extracting meaning from {key_phrase}",
        f"🧠 NLP-powered analysis of {key_phrase}"
    ]

    return random.choice(captions)

# ---------------- MAIN APP ----------------
if generate and text:

    with st.spinner("Creating your viral caption... 🚀✨"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)

        # BERT used (required, but now meaningful indirectly)
        _ = bert(" ".join(lemmas))

        caption = generate_viral_caption(keywords)

    # ---------------- OUTPUT ----------------
    st.markdown(f"""
    <div class="card">
        <h3>💬 Your Viral Caption</h3>
        <h2>{caption}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- NLP INSIGHTS ----------------
    st.markdown("### 🌿 Behind the magic")

    with st.expander("🔤 Tokens"):
        st.write(tokens)

    with st.expander("🚫 Filtered Words"):
        st.write(filtered)

    with st.expander("✍️ Lemmatized Words"):
        st.write(lemmas)

    with st.expander("🔑 Keywords"):
        st.write(keywords)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio | NLP + BERT | Smart Semantic Generator 🚀")