## Carregando arquivo
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data-raw/Manual_PDD.pdf")
docs = loader.load()

## Organizando arquivo para splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

## Criando banco de dados
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma.from_documents(
    documents=all_splits, embedding=embeddings, collection_name="pdd"
)

## Inicializando embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

## Criando vector store
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="pdd", embedding_function=embeddings, persist_directory="data-pdd"
)

## Indexando trechos do arquivo

_ = vector_store.add_documents(all_splits)

## Definindo um prompt para o formato pergunta-resposta
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

## Definindo o estado e os passos para uso da aplicação
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

## Definindo o modelo
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")

## Definindo a aplicação
from langgraph.graph import START, StateGraph


def generate(state: State):
    if state is None or state["context"] is None:
        raise ValueError("Received null state or context")

    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        if messages is None:
            raise ValueError("Received null messages")
        response = llm.invoke(messages)
        if response is None or response.content is None:
            raise ValueError("Received null response or content")
        return {"answer": response.content}
    except Exception as e:
        raise ValueError(f"Encountered an unexpected error: {str(e)}") from e

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

## Executando a aplicação
response = graph.invoke({"question": "Quais dados são disponibilizados na PDD?"})
print(response["answer"])
