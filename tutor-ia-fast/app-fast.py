from fastapi import FastAPI
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from lanarky import LangchainRouter
from starlette.requests import Request
from starlette.templating import Jinja2Templates

app = FastAPI()

llm = ChatOpenAI(
    api_key="ollama",
    model="llama3:latest",
    base_url="http://localhost:11434/v1",
)

chain = ConversationChain(
        # llm=ChatOpenAI(
        #     temperature=0,
        #     streaming=True,
        # ),
        llm=llm,
        verbose=True,
)

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

langchain_router = LangchainRouter(
    langchain_url="/chat", langchain_object=chain, streaming_mode=1
)
langchain_router.add_langchain_api_route(
    "/chat_json", langchain_object=chain, streaming_mode=2
)

langchain_router.add_langchain_api_websocket_route("/ws", langchain_object=chain)
app.include_router(langchain_router)

#uvicorn app-fast:app --reload
