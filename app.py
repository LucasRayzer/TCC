import streamlit as st
from utils import chatbot_Gemini
from streamlit_chat import message

st.set_page_config(page_title='ChatUdesc', page_icon=':books:')
st.header('Tire suas dúvidas institucionais!')

# Verifica se a conversation chain já foi criada para esta sessão
if "conversation" not in st.session_state:
    # Mostra uma mensagem de carregando enquanto a chain é criada pela primeira vez
    with st.spinner("Aguarde, preparando o assistente..."):
        # A primeira chamada a load_vectorStore() é carregada do disco
        # Todas as próximas chamadas serão instantâneas pois pega do cachce
        vector_store = chatbot_Gemini.load_vectorStore()
        
        # Cria a chain de conversação
        st.session_state.conversation = chatbot_Gemini.create_conversation_chain(vector_store)

# Interface
# Inicializa o histórico de mensagens se ele não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens antigas no início
for msg in st.session_state.messages:
    message(msg["content"], is_user=msg["is_user"])

# Input para a nova pergunta do usuário
user_question = st.text_input("Faça uma pergunta...", key="user_input")

if user_question:
    # Adiciona a pergunta do usuário ao histórico e à tela
    st.session_state.messages.append({"content": user_question, "is_user": True})
    message(user_question, is_user=True)
    
    # Processa a pergunta e obtém a resposta do bot
    response = st.session_state.conversation.invoke({"question": user_question})
    answer = response["answer"]
    
    # Adiciona a resposta do bot ao histórico
    st.session_state.messages.append({"content": answer, "is_user": False})
    message(answer, is_user=False)