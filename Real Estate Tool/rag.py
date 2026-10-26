import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from uuid import uuid4
import os
from pathlib import Path

# Sahi chain import jo bina source metadata errors ke kaam karegi
from langchain_classic.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
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

# Constraints
CHUNK_SIZE = 1000
EMBEEDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "realstatetool"
llm = None
vector_store = None

def initialize_components():
    global llm, vector_store
    
    # 🛠️ FIX 1: Stable deployment configuration framework standard mapping
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-latest",  # Sub-dependency mapping standard fallback ID
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
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
    This function scrapes data from dynamic input URLs and stores it in Chroma DB.
    :param urls: list of input URLs from Streamlit inputs
    """
    yield "Initializing Components..."
    initialize_components()
    
    yield "Resetting vectorStore ...."
    vector_store.reset_collection()
    
    yield "Loading data from input URLs..."
    all_docs = []
    
    # User ke diye gaye har link ko iterate karke dynamic scrap karein
    for url in urls:
        if url.strip():
            try:
                # 🛠️ FIX 2: Strategy change to handle direct serverless web loaders
                loader = UnstructuredLoader(
                    web_url=url.strip(),
                    strategy="fast"
                )
                all_docs.extend(loader.load())
            except Exception as e:
                yield f"Warning: Failed to load {url} due to error: {str(e)}"

    if not all_docs:
        yield "Error: No valid data could be retrieved from the provided URLs."
        return

    yield "Split text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=CHUNK_SIZE,
    )
    docs = text_splitter.split_documents(all_docs)

    yield "Add chunks to vector database..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    yield "Done adding docs in vector database!"

def generate_answer(query):
    global llm
    if vector_store is None:
        raise RuntimeError("Vector database is not initialized")

    if llm is None:
        initialize_components()
     
    # 🛠️ FIX 3: Custom map instructions standard parsing bypass config
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        input_key="query"
    )
    
    # Query execute karein
    result = chain.invoke({"query": query})
    
    # Return output string
    return result.get("result", "No answer generated.")
