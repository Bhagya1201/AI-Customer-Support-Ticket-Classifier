
import streamlit as st
import joblib


# --------------------------------------------------
# Load Models
# --------------------------------------------------

queue_model = joblib.load("queue_model.pkl")
queue_tfidf = joblib.load("queue_tfidf.pkl")

priority_model = joblib.load("priority_model.pkl")
priority_tfidf = joblib.load("priority_tfidf.pkl")


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Classifier",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI Customer Support Ticket Classifier")

st.write(
    "An NLP-based machine learning system that automatically "
    "classifies customer support tickets into the appropriate "
    "support queue and priority level."
)

st.divider()


# --------------------------------------------------
# Ticket Input
# --------------------------------------------------

st.subheader("📝 Enter Customer Support Ticket")

ticket = st.text_area(
    "Customer Ticket",
    placeholder=(
        "Example: My payment was deducted twice "
        "for the same transaction."
    ),
    height=180
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Ticket", use_container_width=True):

    if not ticket.strip():

        st.warning("⚠️ Please enter a customer support ticket.")

    elif len(ticket.strip()) < 10:

        st.warning(
            "⚠️ Please enter a more detailed ticket "
            "for a meaningful prediction."
        )

    else:

        # Queue prediction
        queue_text = queue_tfidf.transform([ticket])
        predicted_queue = queue_model.predict(queue_text)[0]

        # Priority prediction
        priority_text = priority_tfidf.transform([ticket])
        predicted_priority = priority_model.predict(priority_text)[0]

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.success(
                f"📂 **Support Queue**\n\n"
                f"{predicted_queue}"
            )

        with col2:

            if predicted_priority == "high":
                st.error(
                    f"🚨 **Priority**\n\n"
                    f"{predicted_priority.capitalize()}"
                )

            elif predicted_priority == "medium":
                st.warning(
                    f"⚠️ **Priority**\n\n"
                    f"{predicted_priority.capitalize()}"
                )

            else:
                st.info(
                    f"🔵 **Priority**\n\n"
                    f"{predicted_priority.capitalize()}"
                )


# --------------------------------------------------
# About Project
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project uses Natural Language Processing (NLP) and
        Machine Learning to automatically classify customer support
        tickets.

        The system performs two predictions:

        • Support Queue — identifies the appropriate department.

        • Priority — predicts whether the ticket has low, medium,
          or high priority.

        Text data is converted into numerical features using
        TF-IDF, followed by Linear Support Vector Machine (SVM)
        classification.
        """
    )


# --------------------------------------------------
# How It Works
# --------------------------------------------------

with st.expander("⚙️ How It Works"):

    st.write(
        """
        1. Customer enters a support ticket.

        2. The ticket text is converted into TF-IDF features.

        3. The Queue Classification model predicts the support
           department.

        4. The Priority Prediction model predicts the ticket
           priority.

        5. Both predictions are displayed in the application.
        """
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

with st.expander("📊 Model Information"):

    st.write("**Queue Classification**")
    st.write("Model: Linear SVM")
    st.write("Test Accuracy: 50.44%")
    st.write("Test Macro F1: 48.50%")

    st.write("")

    st.write("**Priority Prediction**")
    st.write("Model: Linear SVM")
    st.write("Test Accuracy: 59.50%")
    st.write("Test Macro F1: 57.23%")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Customer Support Ticket Classification & "
    "Priority Prediction | NLP + Machine Learning + Streamlit"
)
