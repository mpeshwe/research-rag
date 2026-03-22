import uuid
from langchain.docstore import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragatouille import RAGPretrainedModel
import requests

load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'RAG_Modular_Script'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')  


def load_documents(urls):
    """Loads documents from a list of URLs."""
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        all_docs.extend(loader.load())
    return all_docs

def get_wikipedia_page(title: str):
    """Retrieve the full text content of a Wikipedia page."""
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


def get_summaries(docs, llm_model="gpt-4o-mini"):
    """Generates summaries for a list of documents using an LLM."""
    llm = ChatOpenAI(model=llm_model, temperature=0)
    summaries = []
    for doc in docs:
        summary = llm.invoke(f"Summarize the following document: {doc.page_content[:1000]}...").content # Limit to first 1000 chars for brevity
        summaries.append(summary)
    return summaries

def create_summary_based_retriever(docs, llm_model="gpt-4o-mini"):
    """Creates a summary-based MultiVectorRetriever."""
    # Generate summaries
    summaries = get_summaries(docs, llm_model)

    # The vectorstore to use to index the child chunks (summaries)
    vectorstore = Chroma(collection_name="summaries_collection", embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # Add summaries to vectorstore
    retriever.vectorstore.add_documents(summary_docs)

    # Add original documents to docstore
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    print("Summary-based retriever created successfully.")
    return retriever

def create_colbert_retriever(collection_data, index_name, model_name="colbert-ir/colbertv2.0"):
    """Creates a ColBERT-based RAGatouille retriever."""
    RAG = RAGPretrainedModel.from_pretrained(model_name)
    RAG.index(
        collection=collection_data,
        index_name=index_name,
        max_document_length=180,
        split_documents=True,
    )
    print(f"ColBERT index '{index_name}' created successfully.")
    retriever = RAG.as_langchain_retriever(k=3)
    return retriever

# --- Example Usage --- 

# 1. Summary-Based Indexing
# web_urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"
# ]
# web_docs = load_documents(web_urls)
# summary_retriever = create_summary_based_retriever(web_docs)
# query = "What are the components of an LLM-powered agent?"
# retrieved_summary_docs = summary_retriever.get_relevant_documents(query,n_results=1)
# print("\n--- Summary-Based Retrieval Results ---")
# print(retrieved_summary_docs[0].page_content[0:500])

# 2. COLBERT Indexing
# wikipedia_doc = get_wikipedia_page("Hayao_Miyazaki")
# colbert_retriever = create_colbert_retriever([wikipedia_doc], "Miyazaki-Colbert-Index")
# query = "What animation studio did Miyazaki found?"
# retrieved_colbert_docs = colbert_retriever.invoke(query)
# print("\n--- ColBERT Retrieval Results ---")
# print(retrieved_colbert_docs)