import os 
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

documents = []

pdf_folder = "files"

#embeddding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(path)
        docs = loader.load()

        #metadata for keeping sources at the time of output
        for doc in docs:
            
            text = doc.page_content

            doc.metadata["file_name"] = file
            doc.metadata["page_number"] = doc.metadata["page"] + 1 # in page attribute of meta data we already have page num but it starts form zero.

        documents.extend(docs)#adds each page one by one in documents , here if we add multiple documents they are kinda stacked over one another
print(len(documents))



#function for page by page chunking
def chunk_page(text, chunk_size=500, chunk_overlap=50):
    chunks=[]
    start=0

    while(start<len(text)):
        end = start + chunk_size
        chunk = text[start:end]

        if(end<len(text)):
            last_period=chunk.rfind(".")
            
            if last_period > chunk_size*0.8:
                chunk = chunk[:last_period+1] # +1 in order to include the full stop 
                end = start + last_period + 1
            
        chunks.append(chunk)
        
        start = end - chunk_overlap # resetting the start value to overlap a part of previous chunk
    
    return chunks

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="ml_chunks"
)

chunk_id=0

for doc in documents:
    
    chunks = chunk_page(doc.page_content, chunk_size=500, chunk_overlap=50)
    embeddings = embedding_model.encode(chunks)

    #unique id for each chunk
    ids = [f"chunk_{chunk_id+i}" for i in range(len(chunks))]

    #metadata list 
    metadatas = [
        {
        "file_name":doc.metadata["file_name"],
        "page_number":doc.metadata["page_number"]
        }   
        for _ in chunks
    ]

    collection.add(
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    
    chunk_id+=len(chunks)

print(collection.count())



#query testing 

query = "what is bert"
query_embedding = embedding_model.encode(query)

result = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrived_chunks = result["documents"][0]  
context = "\n\n".join(retrived_chunks)

