import QueryTranslation as QT
from langchain_openai import ChatOpenAI
import QueryConstruction as QC
import indexing as IDX
import os
from dotenv import load_dotenv

load_dotenv("environmentVariables.env")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'RAG_Modular_Script'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') 

def printLine():
    print("\n" + "-"*50 + "\n")

question = "What are the most influential papers on reinforcement learning with words less than 50000?"
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 1. Multi-query generator
multi_query_chain = QT.get_multi_query_chain_only_queries(llm)
print("Multi-query chain initialized with LLM:")
printLine()

# 2. Build index + get retriever
multi_vector_retriever = IDX.indexing()
print("MultiVectorRetriever initialized with retriever type:")
printLine()

# 3. Get vectorstore from it
vectorstore = multi_vector_retriever.vectorstore
print("Vectorstore collection name:", vectorstore._collection.name)
printLine()

# 4. Self-query retriever
self_query_retriever = QC.construct_self_query_retriever(vectorstore, llm)
print("SelfQueryRetriever initialized with retriever type:", type(self_query_retriever))
printLine()

#["query 1","query 2",...]
queries = multi_query_chain.invoke({"question": question})
print("Generated sub-queries:", queries)
printLine()

# calling retriever for each query and taking union of results
all_docs = []

for q in queries:
    docs = self_query_retriever.invoke(q)
    all_docs.extend(docs)
    print(f"Retrieved {len(docs)} documents for query: '{q}'")
printLine()

# get unique docs based on title
unique_docs = {doc.metadata["title"]: doc for doc in all_docs}
final_docs = list(unique_docs.values())

#atlast deserialize the docs before using them
final_docs = IDX.deserialize_docs(final_docs)
print(f"Final retrieved docs for question '{question}':")
printLine()
print(f"Total unique documents retrieved: {len(final_docs)}")
printLine()
for doc in final_docs:
    print(f"""Title: {doc.metadata['title']}, 
          Authors: {doc.metadata['authors']}, 
          Published Year: {doc.metadata['published_year']}, 
          Word Count: {doc.metadata['word_count']}""")
    printLine()