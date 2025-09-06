from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text = """
A aurora costura um ouro pálido pelo beco de vidro.
A cidade boceja em coro de freios e sirenes distantes.
Janelas piscam acordando, uma a uma, como olhos sonolentos.
Tecidos de vapor se erguem dos bueiros, um rio silencioso.
O vapor do café espirala acima do jornal de impressão pálida.
Pedestres desenham luz nas calçadas, apressados, barulhentos com guarda-chuvas.
Ônibus engolem a manhã com seus bocejos estrondosos.
Um pardal pousa em uma viga de aço, inspecionando a malha.
O metrô suspira em algum lugar subterrâneo, um batimento subindo.
O néon ainda brilha nos cantos onde a noite recusou-se a se aposentar.
Um ciclista corta o coro, brilhante de cromo e impulso.
A cidade pigarreia, o ar ficando um pouco menos elétrico.
Sapatos sussurram no concreto, mil pequenos verbos de chegada.
A aurora mantém suas promessas no ritmo calmo de uma metrópole que desperta.

A luz da manhã desce em cascata pelas janelas imponentes de aço e vidro,
lançando sombras geométricas sobre as ruas movimentadas abaixo.
O tráfego flui como rios de metal e luz,
enquanto pedestres atravessam as faixas com propósito.
Cafeterias exalam calor e o aroma de pão fresco,
enquanto os viajantes agarram seus copos como talismãs contra o frio.
Vendedores ambulantes gritam em uma sinfonia de idiomas,
suas vozes se misturam ao zumbido distante da construção.
Pombos dançam entre os pés de trabalhadores apressados,
encontrando migalhas de pães matinais nas calçadas de concreto.
A cidade respira no compasso de um milhão de corações,
cada pessoa carregando sonhos e prazos em igual medida.
Arranha-céus alcançam as nuvens que vagam como algodão,
enquanto, muito abaixo, trens do metrô rugem pelos túneis.
Essa orquestra urbana toca do amanhecer ao entardecer,
uma canção interminável de ambição, luta e esperança.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=70)

parts = splitter.create_documents([long_text])

# for part in parts:
#   print(part.page_content)
#   print("--"*10)

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

chain_sumarize = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

result = chain_sumarize.invoke({"input_documents": parts})

print(result)
print("-----"*5)
print(result["output_text"])