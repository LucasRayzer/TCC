import os
from dotenv import load_dotenv
import streamlit as st
from utils import chatbot
from streamlit_chat import message

# Carrega variáveis do .env
load_dotenv()

st.set_page_config(page_title='ChatUdesc', page_icon=':books:')
st.header('Tire suas dúvidas institucionais!')

# --- Inicialização segura do VectorStore ---
@st.cache_resource
def get_vector_store():
    return chatbot.load_vectorStore()  # Carrega Chroma + embeddings

vectorStore = get_vector_store()

# --- Inicialização do Conversation Chain ---
def get_conversation_chain():
    # Criar uma chain apenas se não existir
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = chatbot.create_conversation_chain(vectorStore)
    return st.session_state.conversation_chain

conversation = get_conversation_chain()

# --- Interface ---
def main():
    user_question = st.text_input("Faça uma pergunta...")

    if user_question:
        response = conversation.invoke({"question": user_question})
        chat_history = response["chat_history"]

        for i, text_message in enumerate(chat_history):
            if i % 2 == 0:
                message(text_message.content, is_user=True, key=f"{i}_user")
            else:
                message(text_message.content, is_user=False, key=f"{i}_bot")


if __name__ == "__main__":
    main()
