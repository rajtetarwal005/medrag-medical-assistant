import streamlit as st
import requests
import os
import uuid

API_URL = "https://medrag-backend-44n5.onrender.com"
# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediBot AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediBot - AI Medical Assistant")

# ---------------------------------------------------------
# Session ID (VERY IMPORTANT 🔥)
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a medical PDF", type=["pdf"])

UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if uploaded_file is not None and "file_uploaded" not in st.session_state:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    with st.spinner("Processing document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
        },
        data={
            "session_id": session_id
        }
    )

    if response.status_code == 200:
        st.success("Document processed and ready!")
        st.session_state.file_uploaded = True
    else:
        st.error("Upload failed. Please try again.")

# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# Chat Input (ONLY AFTER UPLOAD 🔥)
# ---------------------------------------------------------
if "file_uploaded" in st.session_state:
    user_query = st.chat_input("Ask your medical question...")
else:
    st.info("Please upload a PDF first to start chatting.")
    user_query = None

# ---------------------------------------------------------
# Chat Logic
# ---------------------------------------------------------
if user_query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={
                        "query": user_query,
                        "session_id": session_id   # 🔥 important
                    }
                )

                if response.status_code != 200:
                    st.error("Server error. Please try again.")
                else:
                    data = response.json()

                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)

                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.write(f"{i}. {src}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

            except Exception as e:
                st.error(f"Error: {str(e)}")