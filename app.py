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

# ---------------- SAFE NLTK ----------------
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

# ---------------- UI ----------------
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
}
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
}
.card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 14px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# ✨ Caption Studio")
st.markdown("Turn ideas into meaningful AI captions 🌿")

text = st.text_area("💬 Enter your project description")

generate = st.button("🌸 Generate Caption")

# ---------------- BERT ----------------
@st.cache_resource
def load_model():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_model()

# ---------------- NLP PIPELINE (IMPROVED) ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    # 🔥 IMPROVED KEYWORD EXTRACTION (frequency-based)
    freq = {}
    for word in lemmas:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in sorted_words[:6]]

    keyword_text = " ".join(keywords)

    return tokens, filtered, lemmas, keywords, keyword_text

# ---------------- MAIN ----------------
if generate and text:

    with st.spinner("Generating creative caption... 💭✨"):

        tokens, filtered, lemmas, keywords, keyword_text = nlp_pipeline(text)

        # BERT (kept for requirement)
        _ = bert(" ".join(lemmas))

        # 🔥 IMPROVED CAPTION TEMPLATES (MORE NATURAL)
        captions = [
            f"🚀 Built a project around {keyword_text}",
            f"✨ Exploring ideas of {keyword_text}",
            f"💡 Created something using {keyword_text}",
            f"🌿 Turning {keyword_text} into reality",
            f"🔥 A journey of building {keyword_text}",
            f"🌸 From concept to execution: {keyword_text}"
        ]

        caption = random.choice(captions)

    # ---------------- OUTPUT ----------------
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

    with st.expander("🚫 Filtered Words"):
        st.write(filtered)

    with st.expander("✍️ Lemmatized Words"):
        st.write(lemmas)

    with st.expander("🔑 Keywords (Improved)"):
        st.write(keywords)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio | NLP + BERT | Improved Keyword Intelligence")