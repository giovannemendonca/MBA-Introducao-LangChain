from re import X
import re
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import chain
load_dotenv()


@chain
def square(input_dict: dict) -> dict:
  x = input_dict["x"]
  return {"square_result": x * x }


question_template2 = PromptTemplate(
  input_variables=["square_result"],
  template="Me fale sobre o número {square_result}"
)


model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = square | question_template2 | model

result = chain.invoke({"x": 5})

print(result.content)
