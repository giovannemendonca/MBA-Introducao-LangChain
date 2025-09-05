from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

question_template = PromptTemplate(
  input_variables=["name"],
  template="Olá eu sou {name}! Conte-me uma piada sobre meu nome!"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = question_template | model

response = chain.invoke({"name": "Giovanne"})

print(response.content)