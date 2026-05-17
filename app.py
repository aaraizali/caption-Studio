import streamlit as st
import random
import string
import nltk
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Caption Studio",
    page_icon="✨",
    layout="centered"
)

# ---------------- CLEAN DARK UI ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1220, #0f172a, #020617);
    color: #e5e7eb;
}

/* Title */
h1 {
    text-align: center;
    color: white !important;
    font-weight: 800;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
}

/* Input box */
textarea {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 18px rgba(139,92,246,0.4);
}

/* Card */
.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# ✨ Caption Studio")
st.markdown("<div class='subtitle'>Turn your ideas into meaningful captions using AI 🌿</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area("💬 Describe your idea")

# ---------------- BUTTON ----------------
generate = st.button("🌸 Generate Caption")

# ---------------- NLTK SETUP ----------------
@st.cache_resource
def load_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

load_nltk()

# ---------------- BERT MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_model()

# ---------------- NLP PIPELINE ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    # Keywords
    keywords = lemmas[:5]
    keyword_text = " ".join(keywords)

    return tokens, filtered, lemmas, keywords, keyword_text

# ---------------- OUTPUT ----------------
if generate and text:

    with st.spinner("Creating your caption... 💭"):

        tokens, filtered, lemmas, keywords, keyword_text = nlp_pipeline(text)

        # BERT (required)
        _ = bert(" ".join(lemmas))

        captions = [
            f"🌿 Built around {keyword_text}",
            f"✨ A creative idea: {keyword_text}",
            f"💡 Exploring innovation through {keyword_text}",
            f"🚀 Bringing {keyword_text} to life",
            f"🌸 From thought to creation: {keyword_text}"
        ]

        caption = random.choice(captions)

    # ---------------- CAPTION CARD ----------------
    st.markdown(f"""
    <div class="card">
        <h3>💬 Generated Caption</h3>
        <h2>{caption}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- NLP INSIGHTS ----------------
    st.markdown("### 🌿 Behind the magic")

    with st.expander("🔤 Tokens"):
        st.write(tokens)

    with st.expander("🚫 Clean Words (Stopword Removed)"):
        st.write(filtered)

    with st.expander("✍️ Lemmatized Words"):
        st.write(lemmas)

    with st.expander("🔑 Keywords"):
        st.write(keywords)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio | NLP + BERT | Human-friendly AI")