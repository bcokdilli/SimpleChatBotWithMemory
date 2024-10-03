from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a high school chemistry teacher. Your job is to explain chemistry concepts in a clear and engaging way, suitable for high school students. Focus on making complex topics easy to understand, and be patient and supportive when students ask questions. Provide examples and practical applications of chemistry when relevant, and use a friendly, encouraging tone.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model
config = {"configurable": {"session_id": "firstChat"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

if __name__ == "__main__":
    while True:
        user_input = input("\n> ")
        for r in with_message_history.stream(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        ):
            print(r.content, end="")
