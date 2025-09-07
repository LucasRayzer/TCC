import os
from dotenv import load_dotenv

# Carrega variáveis do .env logo no início
load_dotenv()

import streamlit as st
from utils import text
from utils import chatbot
from streamlit_chat import message


def main():
    st.set_page_config(page_title='ChatUdesc', page_icon=':books:')

    st.header('Tire suas dúvidas institucionais!')
    user_question = st.text_input("Faça uma pergunta...")

    # Inicializa o estado da conversa se ainda não existir
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Primeiro carregue e processe seus arquivos antes de perguntar.")
        else:
            response = st.session_state.conversation(user_question)['chat_history']

            for i, text_message in enumerate(response):
                if i % 2 == 0:
                    message(text_message.content, is_user=True, key=str(i)+ "_user")
                else:
                    message(text_message.content, is_user=False, key=str(i)+"_bot")

    with st.sidebar:
        st.subheader('Seus Arquivos')
        pdf_docs = st.file_uploader('Carregue seus arquivos', accept_multiple_files=True)

        if st.button('Processar') and pdf_docs:
            all_files_text = text.process_files(pdf_docs)
            chunks = text.create_text_chunks(all_files_text)

            vectorStore = chatbot.create_vectorStore(chunks)
            # Inicializa a conversa
            st.session_state.conversation = chatbot.create_conversation_chain(vectorStore)
            st.success("Arquivos processados! Agora você já pode fazer perguntas.")


if __name__ == '__main__':
    main()
