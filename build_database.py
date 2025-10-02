import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()
print("API Keys loaded (though not needed for this script).")

def main():
    """Main function to build and save a standard vector database."""
    # IMPORTANT: Place your PDF files in the same folder as this script.
    pdf_files = [
        "c:/Users/javed/Downloads/Nelson-essentials-of-pediatrics.pdf",
        "c:/Users/javed/Downloads/Manual of Neonatal Care 7th.pdf",
        "C:/Users/javed/Downloads/Gomella_5Ed.pdf",
        "C:/Users/javed/Downloads/Gupte-The-Short-Textbook-of-Pediatrics-11th-Ed-2009.pdf",
        "C:/Users/javed/Downloads/Illustrated-textbook-of-pediatrics.pdf",
        "C:/Users/javed/Downloads/document.pdf"
    ]

    if not all(os.path.exists(f) for f in pdf_files):
        print("Error: One or more PDF files were not found. Please ensure they are in the same folder as this script.")
        return
        
    print("Loading PDF documents...")
    all_docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            return
    
    print(f"Loaded {len(all_docs)} pages from PDFs.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(docs)} chunks.")

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("\nCreating FAISS vector database directly from document chunks...")
    

    db = FAISS.from_documents(
        documents=docs,
        embedding=embedding_function
    )
    
    db.save_local("faiss_direct_index")
    print("\nVector database created successfully!")
    print("It has been saved to the 'faiss_direct_index' folder.")
    print("Please update your streamlit_app.py to load from this new folder.")

if __name__ == "__main__":
    main()

