import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

# Загружаем переменные окружения
load_dotenv()

def initialize_chat():
    # Инициализируем модель
    chat = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="mixtral-8x7b-32768"
    )
    
    # Инициализируем память
    memory = ConversationBufferMemory()
    
    # Создаём цепочку разговора
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        verbose=True
    )
    
    return conversation

def main():
    st.title("🤖 Чат-бот прототип")
    
    # Инициализируем сессию
    if "conversation" not in st.session_state:
        st.session_state.conversation = initialize_chat()
    
    # История сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Отображаем историю
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Поле ввода
    if prompt := st.chat_input("Введите сообщение..."):
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Получаем ответ от бота
        with st.chat_message("assistant"):
            response = st.session_state.conversation.predict(input=prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()