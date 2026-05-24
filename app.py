import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model + data
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
captions_data = pickle.load(open("captions_data.pkl", "rb"))

st.set_page_config(page_title="AI Caption Generator", page_icon="✨")

st.title("🔥 AI Caption Generator")
st.write("Turn your ideas into viral captions using ML similarity scoring!")

user_input = st.text_area("💬 Enter your idea:")

if st.button("Generate Caption"):

    if not user_input.strip():
        st.warning("Please enter something!")
    else:
        # Convert input + dataset into vectors
        all_texts = captions_data + [user_input]
        vectors = vectorizer.fit_transform(all_texts)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])

        best_index = similarity.argmax()
        best_score = similarity.max()

        st.success("💬 Best Matching Caption:")
        st.write(captions_data[best_index])

        st.info(f"📊 Similarity Score: {round(best_score * 100, 2)}%")