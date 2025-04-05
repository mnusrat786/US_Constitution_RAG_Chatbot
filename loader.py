
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def load_and_split(file_path, file_type="txt"):
    # Choose the appropriate loader based on the file type
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    # Load the documents
    documents = loader.load()

    # Split the documents into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)