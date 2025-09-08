from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Avalia uma expressão matemática simples e retorna o resultado como string."""
    try:
        result = eval(expression)  # cuidado: apenas exemplo didático!
    except Exception as e:
        return f"Erro: {e}"
    return str(result)

@tool("pesquisa_capital_mock")
def pesquisa_capital_mock(query: str) -> str:
    """Retorna a capital de um país, se ele existir nos dados simulados (mock)."""
    data = {
        "Brasil": "Brasília",
        "França": "Paris",
        "Alemanha": "Berlim",
        "Itália": "Roma",
        "Espanha": "Madri",
        "Estados Unidos": "Washington, D.C."
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"A capital de {country} é {capital}."
    return "Não sei a capital desse país."

llm = ChatOpenAI(model="gpt-5-mini", disable_streaming=True)
tools = [calculator, pesquisa_capital_mock]

prompt = PromptTemplate.from_template(
"""
Responda às seguintes perguntas da melhor forma possível. 
Você tem acesso às seguintes ferramentas.
Use apenas as informações que você obtiver das ferramentas, mesmo que já saiba a resposta.
Se a informação não for fornecida pelas ferramentas, diga que não sabe.

{tools}

Use o seguinte formato (não traduza os marcadores!):

Question: a pergunta que você deve responder  
Thought: seu raciocínio sobre o que fazer  
Action: a ação a ser tomada, deve ser uma das [{tool_names}]  
Action Input: a entrada para a ação  
Observation: o resultado da ação  

... (este bloco pode se repetir N vezes)  
Thought: agora sei a resposta final  
Final Answer: a resposta final para a pergunta original  

Regras:  
- Se você escolher uma Action, NÃO inclua a Final Answer no mesmo passo.  
- Depois de Action e Action Input, pare e aguarde pela Observation.  
- Nunca pesquise na internet. Use apenas as ferramentas fornecidas.  
- Sempre dê a resposta final em português.  

Comece!  

Question: {input}  
Thought:{agent_scratchpad}"""
)

agent_chain = create_react_agent(llm, tools, prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors="Formato inválido. Forneça uma Action com Action Input, ou apenas uma Final Answer.",
    max_iterations=3
)

#print(agent_executor.invoke({"input": "Qual é a capital do Iran?"}))
print(agent_executor.invoke({"input": "Quanto é 10 + 10?"}))