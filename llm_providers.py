from langchain.schema import BaseMessage, ChatGeneration, Generation, AIMessage, ChatResult
from langchain.chat_models.base import BaseChatModel
from langchain_groq import ChatGroq
from cerebras.cloud.sdk import Cerebras
from typing import List, Optional, Dict, Any
from pydantic import PrivateAttr

class BaseLLMProvider:
    """Базовый класс для провайдеров LLM"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_available_models(self) -> List[str]:
        """Возвращает список доступных моделей"""
        raise NotImplementedError
        
    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Создает и возвращает модель"""
        raise NotImplementedError

class GroqProvider(BaseLLMProvider):
    """Провайдер для Groq"""
    def get_available_models(self) -> List[str]:
        return [
            "mixtral-8x7b-32768",
            "llama-3.1-70b-versatile"
        ]
    
    def create_model(self, model_name: str, **kwargs) -> ChatGroq:
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            **kwargs
        )

class CerebrasProvider(BaseLLMProvider):
    """Провайдер для Cerebras"""
    def get_available_models(self) -> List[str]:
        return [
            "llama3.1-8b",
            "llama3.1-70b"
        ]
    
    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        class CerebrasChat(BaseChatModel):
            """Обертка для Cerebras API"""
            model: str
            """Название модели"""
            
            _client: Cerebras = PrivateAttr()
            """Клиент Cerebras API"""
            
            def __init__(self, api_key: str, model: str, **kwargs):
                super().__init__(model=model, **kwargs)
                self._client = Cerebras(api_key=api_key)
                
            @property
            def _llm_type(self) -> str:
                return "cerebras"
            
            def _convert_message_to_cerebras(self, message: BaseMessage) -> Dict[str, str]:
                """Конвертирует сообщение LangChain в формат Cerebras"""
                role_mapping = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system"
                }
                
                role = role_mapping.get(message.type, "user")
                return {
                    "role": role,
                    "content": message.content
                }
                
            def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
                try:
                    # Преобразуем сообщения
                    cerebras_messages = [
                        self._convert_message_to_cerebras(msg)
                        for msg in messages
                    ]
                    
                    # Добавляем системное сообщение если его нет
                    if not cerebras_messages or cerebras_messages[0]["role"] != "system":
                        cerebras_messages.insert(0, {
                            "role": "system",
                            "content": "You are a helpful AI assistant."
                        })
                    
                    print(f"Debug - Messages being sent to Cerebras: {cerebras_messages}")
                    
                    response = self._client.chat.completions.create(
                        messages=cerebras_messages,
                        model=self.model,
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 4096),
                    )
                    
                    message = AIMessage(content=response.choices[0].message.content)
                    generation = ChatGeneration(message=message)
                    
                    return ChatResult(generations=[generation])
                    
                except Exception as e:
                    print(f"Error in Cerebras API call: {str(e)}")
                    raise
                
        return CerebrasChat(api_key=self.api_key, model=model_name, **kwargs)

# Фабрика провайдеров
def get_provider(provider_name: str, api_key: str) -> BaseLLMProvider:
    """Возвращает провайдера по имени"""
    providers = {
        "groq": GroqProvider,
        "cerebras": CerebrasProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
        
    return providers[provider_name](api_key)