import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

captions_data = [
    "Living my best life ✨ #vibes #life",
    "Nature heals everything 🌿 #peace #nature",
    "A dog running in the garden 🐶🌳 #cute #pets",
    "Small moments, big memories 💫 #life",
    "Stay consistent, success will follow 🚀 #motivation",
    "Sunshine and good vibes only ☀️ #happy",
    "Enjoying the little things in life 🌸 #gratitude",
    "Hard work always pays off 💪 #success",
    "Dream big, work hard ✨ #goals",
    "Life is better with animals 🐾 #pets #love"
]

# Save captions
pickle.dump(captions_data, open("captions_data.pkl", "wb"))

# Train TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(captions_data)

# Save model
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")