import os
import sys
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "PDF"

# Load Dokumen PDF
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Split Dokumen menjadi potongan-potongan kalimat (chunk)
def split_documents(documents: list[Document]): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# Buat penyematan (Embedding) menggunakan model ollama
def get_embeddings(): 
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    return embedding

# Buatkan fungsi untuk membuat ID Unik untuk tiap chunk
def create_chunks_id(chunks): 
    # ID Dibuat seperti ini "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Index of chunk

    last_page_id = None
    current_chunk_index = 0

    # Jika Page ID sama dengan Page terakhir
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Jika ID Page masih sama maka increment-kan index dari chunk. jika tidak maka mulai dari 0
        if current_page_id == last_page_id:
            current_chunk_index +=  1
        else : 
            current_chunk_index = 0
        
        # buat ID untuk chunk
        chunk_id = f"{current_page_id}:{current_chunk_index}"

        # Perbarui index dari Current Page ID
        last_page_id = current_page_id

        # Tambahkan ke metadata
        chunk.metadata["id"] = chunk_id
    
    return chunks


# Buat database vektor dengan chroma
def add_to_chroma(chunks: list[Document]):
    # Membuat database kosongan untuk nantinya diisi oleh chunk
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

    # Membuat id untuk Tiap chunks
    chunks_with_ids = create_chunks_id(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"Number of existing documents in DB : {len(existing_ids)}")

    new_chunks = []    
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids) # mengisi database dengan data
        db.persist()

    else :
        print("NO document added")

def clear_database(): 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def query_rag(text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings)
    result = db.similarity_search_with_score(text, k=5)

    return result


response = query_rag("apa itu pendidikan")
print(response)
# if __name__ == "__main__":
#     load_documents()