import uuid

import requests
import streamlit as st

API_URL = "https://notesengine-api-production.up.railway.app/query"
UPLOAD_URL = "https://notesengine-api-production.up.railway.app/upload"

st.set_page_config(page_title="NotesEngine Agentic RAG", layout="wide")
st.title("NotesEngine - Your Personal Agentic RAG Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload documents to append to the KB",
        type=["pdf", "txt", "docx", "pptx"],
        accept_multiple_files=True,
    )

    upload_clicked = st.button("Upload & ingest")

    if upload_clicked:
        if not uploaded_files:
            st.warning("Please choose at least one file.")
        else:
            multipart_files = []
            for f in uploaded_files:
                multipart_files.append(
                    (
                        "files",
                        (f.name, f.getvalue(), f.type or "application/octet-stream"),
                    )
                )

            with st.spinner("Uploading and ingesting..."):
                response = requests.post(
                    UPLOAD_URL,
                    files=multipart_files,
                    data={"session_id": st.session_state.session_id},
                    timeout=600,
                )
                response.raise_for_status()
                upload_data = response.json()

            st.success("Upload complete.")
            if upload_data.get("saved_files"):
                st.write("Saved files:")
                for name in upload_data["saved_files"]:
                    st.write(f"- {name}")

            if upload_data.get("results"):
                st.write("Ingestion results:")
                for item in upload_data["results"]:
                    st.write(
                        f"- {item['file_name']}: {item['status']} "
                        f"({item.get('reason', '')})"
                    )

            if upload_data.get("skipped"):
                st.write("Skipped:")
                for item in upload_data["skipped"]:
                    st.write(
                        f"- {item['file_name']}: {item['status']} "
                        f"({item['reason']})"
                    )

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("New chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

with col2:
    st.caption(f"Session ID: {st.session_state.session_id}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question from your notes")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={
                    "query": user_query,
                    "session_id": st.session_state.session_id,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

        st.markdown(data["answer"])

        if data.get("followup_suggestions"):
            st.subheader("Suggested follow-ups")
            for item in data["followup_suggestions"]:
                st.write(f"- {item}")

        if data.get("contexts"):
            with st.expander("Retrieved contexts"):
                for i, ctx in enumerate(data["contexts"], 1):
                    source = ctx.get("source", "unknown")
                    page = ctx.get("page", "unknown")
                    rerank_score = ctx.get("rerank_score")
                    retrieval_distance = ctx.get("retrieval_distance")
                    st.write(f"**{i}. {source} | Page: {page}**")
                    if retrieval_distance is not None:
                        st.write(f"FAISS distance: {retrieval_distance:.4f}")
                    if rerank_score is not None:
                        st.write(f"Rerank score: {rerank_score:.4f}")
                    st.write(ctx.get("text", ""))
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
