import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Определяем системный промпт
SYSTEM_PROMPT = """Ты - дружелюбный и умный ассистент. 
Твои ответы всегда четкие и по делу.
Если тебе задают технический вопрос - ты даешь примеры кода.
Если не знаешь ответ - честно говоришь об этом.
"""

# Шаблон для разговора
TEMPLATE = """
{system_prompt}

История разговора:
{history}

Человек: {input}
Ассистент:"""

def initialize_chat():
    # Инициализируем модель (можно менять модель здесь)
    chat = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        # Варианты моделей:
        # model_name="mixtral-8x7b-32768"
        model_name="llama-3.1-70b-versatile",
        temperature=0.7,  # Настройка креативности (0.0 - 1.0)
        max_tokens=4096,  # Максимальная длина ответа
    )
    
    # Создаём шаблон промпта
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        partial_variables={"system_prompt": SYSTEM_PROMPT},
        template=TEMPLATE
    )
    
    # Инициализируем память
    memory = ConversationBufferMemory(
        human_prefix="Человек",
        ai_prefix="Ассистент"
    )
    
    # Создаём цепочку разговора с новым промптом
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        prompt=prompt,
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