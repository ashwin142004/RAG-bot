import streamlit as st
import google.generativeai as genai
import chromadb
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection("rag_collection")

# Chunking
def chunk_text(text, max_tokens=300):
    tokenizer = tiktoken.encoding_for_model('gpt-4')
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        if len(tokenizer.encode(' '.join(chunk + [word]))) > max_tokens:
            chunks.append(' '.join(chunk))
            chunk = [word]
        else:
            chunk.append(word)
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

# Chunk insersion
def insert_text_into_chroma(text, source_id="user_upload"):
    chunks = chunk_text(text)
    ids = []
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        embedding = get_gemini_embedding(chunk)
        ids.append(f"{source_id}_{i}")
        embeddings.append(embedding)
        metadatas.append({"text": chunk})

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

# Query 
def query_with_context_gemini(question, top_k=3):
    query_embedding = get_gemini_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['metadatas']
    )
    contexts = [match['text'] for match in results['metadatas'][0]] if results['metadatas'][0] else []
    context_text = "\n\n".join(contexts)

    prompt = f"""Answer the following question based on the context below.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"""

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([prompt])
    return response.text.strip()



def get_knowledge_base_entries():
    all_ids = collection.get(include=["metadatas"])
    return list(zip(all_ids["ids"], all_ids["metadatas"])) if all_ids["ids"] else []

def clear_knowledge_base():
    all_ids = collection.get()
    if all_ids["ids"]:
        collection.delete(ids=all_ids["ids"])
        st.session_state["data_uploaded"] = False
        st.success("Knowledge base has been cleared.")
    else:
        st.info("Knowledge base is already empty.")


st.set_page_config(page_title="📚 RAG Chat", layout="wide")
st.title("RAG Chat")


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "data_uploaded" not in st.session_state:
    st.session_state["data_uploaded"] = False


uploaded_file = st.sidebar.file_uploader("📁 Upload Knowledge Base File", type=["txt"])
if uploaded_file and not st.session_state["data_uploaded"]:
    file_content = uploaded_file.read().decode("utf-8")
    insert_text_into_chroma(file_content, source_id=uploaded_file.name)
    st.sidebar.success(f"'{uploaded_file.name}' added to Knowledge Base.")
    st.session_state["data_uploaded"] = True


st.sidebar.markdown("---")
if st.sidebar.button("🔍 View Knowledge Base"):
    entries = get_knowledge_base_entries()
    if not entries:
        st.sidebar.write("Knowledge base is empty.")
    else:
        for idx, metadata in entries:
            st.sidebar.markdown(f"**ID:** {idx}")
            st.sidebar.code(metadata['text'])

if st.sidebar.button("🗑️ Clear Knowledge Base"):
    clear_knowledge_base()


st.markdown("---")
chat_container = st.container()

for entry in st.session_state.chat_history:
    with chat_container.chat_message(entry["role"]):
        st.markdown(entry["content"])


if st.session_state["data_uploaded"]:
    user_input = st.chat_input("Type your question here...")
    if user_input:
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with chat_container.chat_message("user"):
            st.markdown(user_input)

        
        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = query_with_context_gemini(user_input)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a knowledge base file to start chatting.")