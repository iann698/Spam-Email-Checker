import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from datetime import datetime

# Load data
data = pd.read_csv(r'C:\Users\User\SpamEmailChecker\enron_spam_data.csv')

# Clean and preprocess
data.drop(columns=['Date'], inplace=True, errors='ignore')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data['text'] = data['Subject'].astype(str) + ' ' + data['Message'].astype(str)
data['label'] = data['Spam/Ham'].map({'ham': 0, 'spam': 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <h1 style="text-align: center; color: #4B9CD3;">🛡️ AI Spam Email Detector</h1>
    <p style="text-align: center; color: gray;">Using Logistic Regression & TF-IDF</p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.markdown(f"<h4 style='text-align: center;'>Model Accuracy: <span style='color: #28A745;'>{accuracy*100:.2f}%</span></h4>", unsafe_allow_html=True)

email_text = st.text_area("Paste the email content here:")

if st.button("Check Email"):
    email_text = email_text.strip()
    if len(email_text) < 10:
        st.warning("⚠️ Please enter a full email text for reliable classification!")
    else:
        features = vectorizer.transform([email_text])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("🚫 This email is likely Spam.")
            st.warning(
                f"""
                💡 Tips:
                ⚠️ Be careful with links in this email.  
                ⚠️ Avoid opening attachments in this email.  
                ⚠️ Do not reply to this email unless you are certain of the sender.  
                """
            )
        else:
            st.success("✅ This email looks Legitimate.")
            st.info(
                """
                💡 Tips:  
                - Always verify the sender before clicking links.  
                - Keep your software and antivirus up to date.  
                - Avoid sharing sensitive information unless necessary.  
                """
            )
# Footer
current_year = datetime.now().year
st.markdown(
    f"""
    <hr>
    <p style='text-align: center; color: gray; font-size: 12px;'>
    Developed by Ian © {current_year}. All rights reserved.
    </p>
    """,
    unsafe_allow_html=True
)