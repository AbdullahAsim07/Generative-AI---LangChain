# from uiuid jo jab bhi koi action perform hoga toh us ko aik id deti hai
from uuid import uuid4
import os
# jab bhi ap na path lene ka liye us e hota hai
from pathlib import Path
# ager ap na chain ko chain sa jorna hai toh retterival use hota hai
from langchain_classic.chains import RetrievalQAWithSourcesChain
# yeh document loader ka liye use hota hai 
from langchain_classic.schema import BaseRetriever
from langchain_community.document_loaders import UnstructuredURLLoader
# yeh text splitter ka liye 
from langchain_text_splitters import RecursiveCharacterTextSplitter
# yeh chroma kaliye 
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def load_dotenv():
    """Load KEY=VALUE entries from a .env file beside this script."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


load_dotenv()

# contraints
# ap hum na batana hai ka chunks ka size kitna hoo
CHUNK_SIZE=1000
EMBEEDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR=Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME="realstatetool"
llm=None
vector_store=None
# is ma total 3 function hoo ga sab sa pehle initialize , phir process aur genrate  hoo ga
# sab sa pehle intialize ka hoo ga 
def initialize_components():
   # dono ma sa kesi aik ma sa dekhe ga
   global llm, vector_store
   # ager llm none nhi hai toh is ma sa dekho
   if llm is None:
      llm = ChatGroq(model=" llama-3.1-8b-instant", temperature=0.2, max_tokens=300)
   # ager us ko mil llm sa toh theek hai phir embedd kare ga ager nhi mil toh wo is ko vector store ma dekhe ga
   if vector_store is None:
      ef = HuggingFaceEmbeddings(
         model_name=EMBEEDING_MODEL,
         model_kwargs={"trust_remote_code": True},
      )
      vector_store = Chroma(
         collection_name=COLLECTION_NAME,
         embedding_function=ef,
         persist_directory=str(VECTORSTORE_DIR),
      )

def process_urls(urls):
    """
    This function scrapes data from URLs and stores it in a vector database.
    :param urls: input URLs
    """
    # sab sa pehle initialize karo 
    yield "Initializing Components"
    initialize_components()
    # vectorstore ki collection ko reset karo 
    yield "Resetting vectorStore ...."
    vector_store.reset_collection()
    # unstructured url ko load karo aur data ma save karo 
    yield "Loading data ..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # text ko recursive texte spliiter ka use sa chunks ma diviude karo 
    yield "Split text into chunks"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=CHUNK_SIZE,
    )
    # us ka baad saare chunks ko docs ma save karo 
    docs = text_splitter.split_documents(data)

    yield "Add chunks to vector database"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    yield "Done adding docs in vector database ..."

def generate_answer(query):
    # ager vector store ma quwey ka nawer nhi para hua hai toh llm phir bhi answer dega
    if vector_store is None:
        raise RuntimeError("vector database is not initialized")

    # initialize llm if needed
    if llm is None:
        initialize_components()

    # ager vector store ma data nhi para hua hai toh wo chain ka zariye retrieval sources llm sa call kar dega 
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
    )
    result=chain.invoke({"question":query},return_only_output=True)
    # jo answer ma source ay ga us ko wo apka answer hoo ga
    sources = result.get("sources", "")
    return result["answer"], sources
