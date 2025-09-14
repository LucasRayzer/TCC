import os
from dotenv import load_dotenv
import streamlit as st
from utils import text
from utils import chatbot
import tempfile
from streamlit_chat import message


# Carrega variáveis do .env antes de executar
load_dotenv()

def main():
    st.set_page_config(page_title='ChatUdesc', page_icon=':books:')
    st.header('Tire suas dúvidas institucionais!')

    user_question = st.text_input("Faça uma pergunta...")

    # Inicializa o estado da conversa se ainda não existir
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Primeiro carregue e processe seus arquivos antes de perguntar.")
        else:
            docs = st.session_state.conversation.retriever.get_relevant_documents(user_question)
            st.write("Docs recuperados:", [d.page_content[:200] for d in docs])
            # usando invoke
            response = st.session_state.conversation.invoke({"question": user_question})
            chat_history = response["chat_history"]

            for i, text_message in enumerate(chat_history):
                if i % 2 == 0:
                    message(text_message.content, is_user=True, key=f"{i}_user")
                else:
                    message(text_message.content, is_user=False, key=f"{i}_bot")

    with st.sidebar:
        st.subheader("Seus Arquivos")
        pdf_docs = st.file_uploader("Carregue seus arquivos", accept_multiple_files=True)

        if st.button("Processar") and pdf_docs:
            chunks = []
            for file in pdf_docs:
                if file.name.endswith(".md"):
                    # salva temporário e carrega
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                        tmp.write(file.getbuffer())
                        temp_path = tmp.name
                    chunks.extend(chatbot.load_markdown_chunks(temp_path))
                else:
                    # mantém PDF pelo fluxo atual
                    all_files_text = text.process_files([file])
                    chunks.extend(text.create_text_chunks(all_files_text))

            vectorStore = chatbot.create_vectorStore(chunks)
            st.session_state.conversation = chatbot.create_conversation_chain(vectorStore)
            st.success("Arquivos processados! Agora você já pode fazer perguntas.")

if __name__ == "__main__":
    main()
