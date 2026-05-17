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

# ---------------- NLP PIPELINE ----------------
def nlp_pipeline(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    filtered = [w for w in tokens if w not in stop_words]

    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    freq = {}
    for w in lemmas:
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    BAD_WORDS = {
        "text", "student", "often", "using", "make",
        "build", "created", "project", "work"
    }

    keywords = []
    for w, _ in sorted_words:
        if w not in BAD_WORDS:
            keywords.append(w)

    return tokens, filtered, lemmas, keywords[:5]

# ---------------- SMART CATEGORY DETECTION ----------------
def detect_category(text, keywords):

    text = text.lower()

    tech = {"ai", "python", "code", "app", "streamlit", "model", "data", "ml", "system"}
    study = {"study", "exam", "student", "learn", "learning", "stress", "school"}
    motivation = {"life", "dream", "goal", "success", "journey", "grow"}

    score_tech = sum(1 for w in keywords if w in tech or w in text)
    score_study = sum(1 for w in keywords if w in study or w in text)
    score_motivation = sum(1 for w in keywords if w in motivation or w in text)

    if score_tech >= score_study and score_tech >= score_motivation:
        return "tech"
    elif score_study >= score_tech and score_study >= score_motivation:
        return "study"
    else:
        return "motivation"

# ---------------- VIRAL CAPTION ENGINE ----------------
def generate_viral_caption(keywords, category):

    key_phrase = " ".join(keywords[:4])

    tech_captions = [
        f"🚀 Built a next-gen system using {key_phrase}",
        f"💻 Turning code into innovation with {key_phrase}",
        f"⚡ Engineering ideas into reality: {key_phrase}",
        f"🔥 AI-powered build using {key_phrase}",
        f"🌐 From logic to launch: {key_phrase}"
    ]

    study_captions = [
        f"📚 Learning journey powered by {key_phrase}",
        f"🧠 Turning stress into growth: {key_phrase}",
        f"🌱 Every step builds strength with {key_phrase}",
        f"💡 Studying smarter using {key_phrase}",
        f"🎯 Progress over perfection: {key_phrase}"
    ]

    motivation_captions = [
        f"🌟 Life moves forward with {key_phrase}",
        f"🔥 Growth comes from {key_phrase}",
        f"💪 Building dreams with {key_phrase}",
        f"🚀 Keep pushing with {key_phrase}",
        f"✨ Turning ideas into reality: {key_phrase}"
    ]

    if category == "tech":
        return random.choice(tech_captions)
    elif category == "study":
        return random.choice(study_captions)
    else:
        return random.choice(motivation_captions)

# ---------------- MAIN APP ----------------
if generate and text:

    with st.spinner("Creating your viral caption... 🚀✨"):

        tokens, filtered, lemmas, keywords = nlp_pipeline(text)

        # BERT kept (requirement only)
        _ = bert(" ".join(lemmas))

        category = detect_category(text, keywords)
        caption = generate_viral_caption(keywords, category)

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
st.markdown("💜 Caption Studio | NLP + BERT | Smart Viral Generator 🚀")