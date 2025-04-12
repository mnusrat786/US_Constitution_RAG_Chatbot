
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(chunks):
    # Initialize the embedding model
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create and return the FAISS vector store
    return FAISS.from_documents(chunks, embedding)







# from langchain.vectorstores import FAISS
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS 


# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import GoogleGenerativeAIEmbeddings


# def create_vector_store(chunks):
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return FAISS.from_documents(chunks, embedding)
