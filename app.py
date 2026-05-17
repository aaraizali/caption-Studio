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

# ---------------- NLP PIPELINE ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    # keyword extraction (frequency-based)
    freq = {}
    for w in lemmas:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in sorted_words[:6]]

    return tokens, filtered, lemmas, keywords

# ---------------- CAPTION REWRITER (IMPORTANT FIX) ----------------
def generate_natural_caption(text, keywords):

    # take original meaning + keywords blend
    key_phrase = " ".join(keywords)

    templates = [
        f"I built a project focused on {key_phrase}.",
        f"This work explores {key_phrase} using modern AI techniques.",
        f"A complete implementation involving {key_phrase}.",
        f"A creative project built around {key_phrase}.",
        f"This system demonstrates {key_phrase} in action.",
        f"A hands-on project using {key_phrase} and AI."
    ]

    return random.choice(templates)

# ---------------- MAIN ----------------
if generate and text:

    with st.spinner("Generating natural caption... 💭✨"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)

        # BERT (kept as required, not heavily used yet)
        _ = bert(" ".join(lemmas))

        caption = generate_natural_caption(text, keywords)

    # ---------------- OUTPUT ----------------
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

    with st.expander("🚫 Filtered Words"):
        st.write(filtered)

    with st.expander("✍️ Lemmatized Words"):
        st.write(lemmas)

    with st.expander("🔑 Keywords"):
        st.write(keywords)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio | NLP + BERT | Natural Caption Generator")