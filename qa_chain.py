
from langchain.chains import RetrievalQA

def create_qa_chain(llm, vector_store):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
