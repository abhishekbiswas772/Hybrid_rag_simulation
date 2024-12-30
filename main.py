from rag_pipeline.rag_langchain import RagHandler
import streamlit as st

rag_handler = RagHandler()
st.set_page_config(
        page_title="Basic QA",
)
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about any queries releted to pdf..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    query = {"query": prompt}
    with st.chat_message("assistant"):
        try:
            response = rag_handler.answer_question(
                question=query["query"]
            )
            st.markdown(response)
                    
        except ValueError as e:
            st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": response})




