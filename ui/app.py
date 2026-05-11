import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.title("NotesEngine")

query = st.text_input("Ask a question")

if st.button("Search") and query:
    response = requests.post(
        API_URL,
        json={"query": query}
    )

    data = response.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Retrieved Contexts")

    for i, ctx in enumerate(data["contexts"], 1):
        st.write(f"### Context {i}")
        st.write(ctx)
