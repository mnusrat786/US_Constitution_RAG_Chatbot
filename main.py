import gradio as gr
from loader import load_and_split
from embedder import create_vector_store
from model import load_llm
from qa_chain import create_qa_chain

# Load the PDF
print("📄 Loading and processing PDF...")
chunks = load_and_split("constitution.pdf", file_type="pdf")
print(f"Total chunks loaded: {len(chunks)}")

vector_store = create_vector_store(chunks)
llm = load_llm()
qa_chain = create_qa_chain(llm, vector_store)

# Chat function using OpenAI-style messages
def chat_with_pdf(message, history):
    try:
        print(f"User: {message}")
        result = qa_chain.invoke({"query": message})
        answer = result["result"]
        print(f"Bot: {answer}")
        return {"role": "assistant", "content": answer}
    except Exception as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}

# Gradio 5.22.0 compatible
gr.ChatInterface(
    fn=chat_with_pdf,
    title="Chat with U.S. Constitution",
    theme="soft",
    type="messages"  # Use "messages" instead of "chat"
).launch()
