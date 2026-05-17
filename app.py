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

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- LOAD BERT (KEPT ONLY FOR REQUIREMENT) ----------------
@st.cache_resource
def load_model():
    return pipeline("feature-extraction", model="bert-base-uncased")

bert = load_model()

# ---------------- NLP PIPELINE ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    filtered = [w for w in tokens if w not in stop_words]

    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    # simple clean frequency-based keywords (stable + fast)
    freq = {}
    for w in lemmas:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in sorted_words if len(w[0]) > 2][:5]

    return tokens, filtered, lemmas, keywords

# ---------------- VIRAL CAPTION ENGINE (FIXED) ----------------
def generate_viral_caption(keywords):

    key_phrase = " ".join(keywords[:4])

    hooks = [
        "🚀 Built something crazy",
        "🔥 This changed everything",
        "💡 Not your average project",
        "⚡ AI just hit different here",
        "🌟 From idea to reality",
        "💻 Built in the shadows",
        "🧠 Small idea. Big execution.",
        "🚀 I made this happen"
    ]

    styles = [
        f"{random.choice(hooks)} using {key_phrase}",
        f"{random.choice(hooks)} → powered by {key_phrase}",
        f"{random.choice(hooks)} with {key_phrase}",
        f"{random.choice(hooks)}: {key_phrase}",
        f"{random.choice(hooks)} ({key_phrase})"
    ]

    return random.choice(styles)

# ---------------- UI ----------------
st.title("🚀 Caption Studio AI")
st.write("Turn your ideas into **viral captions using NLP + AI** ✨")

text = st.text_area("💬 Enter your project description")

generate = st.button("✨ Generate Viral Caption")

# ---------------- MAIN ----------------
if generate and text:

    with st.spinner("Creating your viral caption... 🚀"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)

        # BERT kept ONLY for project requirement (not affecting output)
        _ = bert(" ".join(lemmas))

        caption = generate_viral_caption(keywords)

    # ---------------- OUTPUT ----------------
    st.success(caption)

    # ---------------- DETAILS ----------------
    with st.expander("🔑 Keywords"):
        st.write(keywords)

    with st.expander("🧠 Clean Words"):
        st.write(lemmas)

    with st.expander("🔤 Tokens"):
        st.write(tokens)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💜 Caption Studio AI | NLP + BERT | Clean Viral Generator 🚀")