### 1

# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(
#     api_key=api_key
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )
# print(chat_completion.choices[0].message.content)

### 2 

# import getpass
# import os

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# from langchain.chat_models import init_chat_model

# llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


### 3

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import

# Initialize your embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create your vector store
vector_store = Chroma(
    persist_directory="path/to/your/directory",
    embedding_function=embeddings
)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# Definindo o modelo
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key
)

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )
# print(chat_completion.choices[0].message.content)

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


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
