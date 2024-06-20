#instalar python 3.10
/opt/homebrew/bin/python3.10 -m venv venv

#embeddings
ollama pull mxbai-embed-large

#instalar dependencias
pip install -r requirements.txt

#DICA DE COMO USAR O CHATOPENAI COM ollama
llm = ChatOpenAI(
    api_key="ollama",
    model="llama3:latest",
    base_url="http://localhost:11434/v1",
)