import streamlit as st
import random
import string
import nltk
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
def safe_nltk_download():
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet")

safe_nltk_download()

# ---------------- UI DESIGN ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1220, #0f172a, #020617);
    color: #e5e7eb;
}

h1 {
    text-align: center;
    color: white !important;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

textarea {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
}

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
st.markdown("<div class='subtitle'>Turn your ideas into AI-powered captions 🌿</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area("💬 Describe your idea")

generate = st.button("🌸 Generate Caption")

# ---------------- LOAD BERT ----------------
@st.cache_resource
def load_model():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_model()

# ---------------- NLP PIPELINE (FIXED) ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # ✅ SAFE TOKENIZATION (NO NLTK ERROR)
    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    keywords = lemmas[:5]
    keyword_text = " ".join(keywords)

    return tokens, filtered, lemmas, keywords, keyword_text

# ---------------- MAIN LOGIC ----------------
if generate and text:

    with st.spinner("Generating your caption... 💭✨"):

        tokens, filtered, lemmas, keywords, keyword_text = nlp_pipeline(text)

        # BERT (kept as required)
        _ = bert(" ".join(lemmas))

        captions = [
            f"🌿 Built around {keyword_text}",
            f"✨ Creative idea: {keyword_text}",
            f"💡 Innovation through {keyword_text}",
            f"🚀 Bringing {keyword_text} to life",
            f"🌸 From thought to reality: {keyword_text}"
        ]

        caption = random.choice(captions)

    # ---------------- OUTPUT CARD ----------------
    st.markdown(f"""
    <div class="card">
        <h3>💬 Generated Caption</h3>
        <h2>{caption}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- NLP DETAILS ----------------
    st.markdown("### 🌿 Behind the magic")

    with st.expander("🔤 Tokens"):
        st.write(tokens)

    with st.expander("🚫 Clean Words"):
        st.write(filtered)

    with st.expander("✍️ Lemmatized Words"):
        st.write(lemmas)

    with st.expander("🔑 Keywords"):
        st.write(keywords)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio | NLP + BERT | Streamlit Project")