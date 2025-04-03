import streamlit as st
import joblib
import re
import string
import nltk
# Ensure necessary NLTK data is available
# You might need to run nltk.download('stopwords') once locally if you haven't
try:
st.set_page_config(page_title="Phishing Detector", layout="wide")

# --- Configuration ---
MODEL_PATH = 'phishing_svm_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

# --- Load Model and Vectorizer ---
# Use cache_resource for efficiency: loads only once per session/app load
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Required file not found. Ensure '{MODEL_PATH}' and '{VECTORIZER_PATH}' are present in the app's root directory.")
        # In a deployed app, st.stop() might be too abrupt. Returning None forces checks later.
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the model or vectorizer: {e}")
        return None, None

model, vectorizer = load_resources()

# --- Text Preprocessing Function (MUST MATCH TRAINING) ---
# stop_words = set(stopwords.words('english')) # Make sure this matches training if you used stopwords

def preprocess_text(text):
    """Cleans and preprocesses text."""
    if not isinstance(text, str): return "" # Basic type check
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\S+@\S+', '', text) # Remove email addresses
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = text.strip() # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace

    # Optional: Remove stopwords (ensure consistency with training)
    # try:
    #     tokens = text.split()
    #     text = ' '.join([word for word in tokens if word not in stop_words])
    # except NameError: # Handle case where stopwords download failed
    #     print("Stopwords not defined, skipping removal.")
    #     pass

    return text

# --- Streamlit App User Interface ---

st.title("🎣 Phishing Email Detector")
st.write("Enter the text content of an email below to check if it might be a phishing attempt.")
st.write("*(Uses a Linear SVM model trained on TF-IDF features)*")

# Use a unique key for the text_area
email_text = st.text_area("Paste Email Text Here:", height=250, key="email_input_area")

# Use a unique key for the button
if st.button("Check Email", key="check_email_button"):

    # Check if resources loaded successfully before proceeding
    if model is None or vectorizer is None:
        st.error("Model or Vectorizer not loaded. Cannot perform prediction. Check logs.")
    elif email_text:
        # 1. Preprocess the input text
        processed_text = preprocess_text(email_text)
        st.write("---") # Separator
        st.write("**Processed Text Snippet (for debugging):**")
        st.code(processed_text[:500] + ('...' if len(processed_text) > 500 else '')) # Show first 500 chars

        if processed_text: # Check if anything remains after processing
            # 2. Vectorize the preprocessed text
            try:
                text_vector = vectorizer.transform([processed_text])
            except Exception as e:
                st.error(f"Error during text vectorization: {e}")
                st.stop() # Stop this run if vectorization fails

            # 3. Make prediction
            try:
                prediction = model.predict(text_vector)
                # 4. Display the result
                st.subheader("Prediction Result:")
                # IMPORTANT: Ensure 'Phishing Email' matches the exact label string from your dataset
                if prediction[0] == 'Phishing Email':
                    st.error(f"🚨 This looks like a **Phishing Email**!")
                else:
                    st.success(f"✅ This looks like a **Safe Email**.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        else:
            # Handles case where input becomes empty after preprocessing
            st.warning("Input text seems empty after cleaning. Please provide more substantial text.")
    else:
        # Handles case where user clicks button with no text entered
        st.warning("Please paste some email text into the box above before clicking 'Check Email'.")

# Add a sidebar with information
st.sidebar.header("About")
st.sidebar.info("""
    This application uses a Machine Learning model (Linear SVM) trained on text features (TF-IDF) to predict whether an email is potentially phishing or safe.

    **Disclaimer:** This is a demonstration tool based on a specific dataset. It may not catch all phishing emails and might occasionally misclassify safe emails. **Always exercise caution** with emails, especially those asking for personal information or containing suspicious links/attachments.
""")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
    1. Paste the full text content of the email you want to check into the main text area.
    2. Click the "Check Email" button.
    3. View the prediction result below the button.
""")
