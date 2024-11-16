import streamlit as st
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from llm_providers import get_provider  # Импортируем нашу фабрику провайдеров

load_dotenv()

# Определяем системный промпт
SYSTEM_PROMPT = """Ты - дружелюбный и умный ассистент. 
Твои ответы всегда четкие и по делу.
Если тебе задают технический вопрос - ты даешь примеры кода.
Если не знаешь ответ - честно говоришь об этом.
"""

def initialize_chat(provider_name: str, model_name: str):
    # Получаем API ключ в зависимости от провайдера
    api_key_map = {
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY"
    }
    
    # Получаем API ключ и проверяем его наличие
    api_key = os.getenv(api_key_map[provider_name])
    if not api_key:
        raise ValueError(f"API key for {provider_name} not found in environment variables")
    
    # Получаем провайдера и создаем модель
    provider = get_provider(provider_name, api_key)
    chat = provider.create_model(
        model_name=model_name,
        temperature=0.7,
        max_tokens=4096
    )
    
    # Создаем новый промпт с использованием ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),  # Явно добавляем системное сообщение
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Создаем цепочку с использованием RunnableWithMessageHistory
    chain = prompt | chat
    
    return chain

def get_message_history():
    """Возвращает историю сообщений в формате LangChain"""
    history = []
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

def main():
    st.title("🤖 Чат-бот прототип")
    
    # Проверяем наличие необходимых API ключей
    for env_var in ["GROQ_API_KEY", "CEREBRAS_API_KEY"]:
        if not os.getenv(env_var):
            st.error(f"Ошибка: {env_var} не найден в переменных окружения!")
            return
    
    # Добавляем селекторы для провайдера и модели
    providers = {
        "groq": ["mixtral-8x7b-32768", "llama-3.1-70b-versatile"],
        "cerebras": ["llama3.1-8b", "llama3.1-70b"]
    }
    
    # Селектор провайдера в сайдбаре
    provider_name = st.sidebar.selectbox(
        "Выберите провайдера:",
        list(providers.keys())
    )
    
    # Селектор модели в сайдбаре
    model_name = st.sidebar.selectbox(
        "Выберите модель:",
        providers[provider_name]
    )
    
    # Инициализируем или переинициализируем сессию при смене провайдера/модели
    if ("chain" not in st.session_state or 
        st.session_state.get("current_provider") != provider_name or
        st.session_state.get("current_model") != model_name):
        
        st.session_state.chain = initialize_chat(provider_name, model_name)
        st.session_state.current_provider = provider_name
        st.session_state.current_model = model_name
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
            response = st.session_state.chain.invoke(
                {"input": prompt, "history": get_message_history()}
            )
            content = response.content if hasattr(response, 'content') else str(response)
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})

if __name__ == "__main__":
    main()