import streamlit as st
import pickle

# ✅ Load trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review, and the model will predict whether it's **Positive** or **Negative**.")

# ✅ User input
review = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review.strip():
        prediction = model.predict([review])[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review before submitting.")
