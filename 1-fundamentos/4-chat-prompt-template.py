from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

system = ("system", "You are an assistant that answers questions in a {style} style.")
user = ("user", "{question}")

chat_promp = ChatPromptTemplate([system, user])

messages = chat_promp.format_messages(style="funny", question="Who is Alan Turing?")

for message in messages:
  print(f"{message.type}: {message.content}")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

response = llm.invoke(messages)

print(response.content)

