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

    # Inicializa o estado da conversa
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Pergunta do usuário
    user_question = st.text_input("Faça uma pergunta...")

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Carregue o índice antes de perguntar.")
        else:
            response = st.session_state.conversation.invoke({"question": user_question})
            chat_history = response["chat_history"]

            for i, text_message in enumerate(chat_history):
                if i % 2 == 0:
                    message(text_message.content, is_user=True, key=f"{i}_user")
                else:
                    message(text_message.content, is_user=False, key=f"{i}_bot")

    # Sidebar só carrega índice pronto
    with st.sidebar:
        st.subheader("Configuração")
        if st.button("Carregar índice"):
            vectorStore = chatbot.load_vectorStore()
            st.session_state.conversation = chatbot.create_conversation_chain(vectorStore)
            st.success("Índice carregado! Agora você já pode fazer perguntas.")
            
if __name__ == "__main__":
    main()
