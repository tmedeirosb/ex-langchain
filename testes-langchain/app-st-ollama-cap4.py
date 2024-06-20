import streamlit as st

from langchain.llms import Ollama
from langchain.agents import initialize_agent, AgentType  # Certifique-se de importar initialize_agent e AgentType

#llm = Ollama(model='splitpierre/bode-alpaca-pt-br:13b-Q4_0')
llm = Ollama(model='llama3:latest')

#uso simples do modelo
prompt = "Qual a capital do rio grande do norte?"
res = llm(prompt)
st.write(res)

#fact-checking
from langchain.chains import LLMCheckerChain

text = "Que tipo de mamífero põe os maiores ovos?"
#checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)
#res = checker_chain.run(text)
#st.write(res)

#Summarizing
prompt = """
Summarize this text in one sentence:

{text} 
"""
#summary = llm(prompt.format(text=text))
#st.write(summary)

# from langchain_decorators import llm_prompt

# @llm_prompt
# def summarize(text:str, length="short") -> str:
#     """
#     Summarize this text in {length} length:
#     {text}
#     """
#     return

# summary = summarize(text="let me tell you a boring story from when I wasy oung...")
# st.write(summary)

#prompts template
from langchain import PromptTemplate
from langchain.schema import StrOutputParser

prompt = PromptTemplate.from_template(
    "Summarize this text: {text}?"
)
runnable = prompt | llm | StrOutputParser()
summary = runnable.invoke({"text": text})
st.write(summary)

#chain of density
template = """Article: { text }
You will generate increasingly concise, entity-dense summaries of the
above article.
Repeat the following 2 steps 5 times.
Step 1. Identify 1-3 informative entities (";" delimited) from the article
which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every
entity and detail from the previous summary plus the missing entities.
A missing entity is:
- relevant to the main story,
- specific yet concise (5 words or fewer),
- novel (not in the previous summary),
- faithful (present in the article),
- anywhere (can be located anywhere in the article).
Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly
non-specific, containing little information beyond the entities marked
as missing. Use overly verbose language and fillers (e.g., "this article
discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and
make space for additional entities.
- Make space with fusion, compression, and removal of uninformative
phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained,
i.e., easily understood without the article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made,
add fewer new entities.
Remember, use the exact same number of words for each summary.
Answer in JSON. The JSON should be a list (length 5) of dictionaries whose
keys are "Missing_Entities" and "Denser_Summary".
"""

#map reduce
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
pdf_file_path = "data/Improving Factuality and Reasoning in Language.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

chain = load_summarize_chain(llm, chain_type="map_reduce")
res = chain.run(docs)

st.write(res)

