from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil que responde com uma piada curta sempre que possível."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.9)

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    return {"input": payload.get("input",""), "history": trimmed}

prepare = RunnableLambda(prepare_inputs)
chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history"
)

config = {"configurable": {"session_id": "demo-session"}}

# Interações em português
resp1 = conversational_chain.invoke({"input": "Meu nome é Giovanne Mendonça. Responda apenas com 'OK' e não mencione meu nome."}, config=config)
print("Assistente:", resp1.content)

resp2 = conversational_chain.invoke({"input": "Conte-me um fato curioso em uma frase. Não mencione meu nome."}, config=config)
print("Assistente:", resp2.content)

resp3 = conversational_chain.invoke({"input": "Qual é o meu nome?"}, config=config)
print("Assistente:", resp3.content)
