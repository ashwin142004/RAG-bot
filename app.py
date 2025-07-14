import streamlit as st
import google.generativeai as genai
import chromadb
import tiktoken
import os
from dotenv import load_dotenv

# === Configure Gemini ===
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# === Initialize ChromaDB ===
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection("rag_collection")

# === Chunking Function ===
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

# === Gemini Embedding ===
def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

# === Insert Chunks into ChromaDB ===
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

# === Query ChromaDB + Gemini ===
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


# === View Knowledge Base ===
def get_knowledge_base_entries():
    all_ids = collection.get(include=["metadatas"])
    return list(zip(all_ids["ids"], all_ids["metadatas"])) if all_ids["ids"] else []

# === Delete Knowledge Base ===
def clear_knowledge_base():
    all_ids = collection.get()
    if all_ids["ids"]:
        collection.delete(ids=all_ids["ids"])
        st.session_state["data_uploaded"] = False
        st.success("Knowledge base has been cleared.")
    else:
        st.info("Knowledge base is already empty.")

# === Streamlit UI ===
st.title("📚 Gemini + ChromaDB RAG App")

if "data_uploaded" not in st.session_state:
    st.session_state["data_uploaded"] = False

# === Upload Section ===
uploaded_file = st.file_uploader("Upload a text file to add to Knowledge Base", type=["txt"])
if uploaded_file and not st.session_state["data_uploaded"]:
    file_content = uploaded_file.read().decode("utf-8")
    insert_text_into_chroma(file_content, source_id=uploaded_file.name)
    st.success(f"File '{uploaded_file.name}' has been added to the Knowledge Base.")
    st.session_state["data_uploaded"] = True

# === Knowledge Base Options ===
st.markdown("---")
st.subheader("🗂 Knowledge Base Options")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 View Knowledge Base"):
        entries = get_knowledge_base_entries()
        if not entries:
            st.write("Knowledge base is empty.")
        else:
            for idx, metadata in entries:
                st.markdown(f"**ID:** {idx}")
                st.code(metadata['text'])

with col2:
    if st.button("🗑️ Clear Knowledge Base"):
        clear_knowledge_base()
        st.success("Knowledge base has been cleared.")

# === Question-Answer Section ===
if st.session_state["data_uploaded"]:
    st.markdown("---")
    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_question.strip() != "":
            with st.spinner("Generating answer..."):
                answer = query_with_context_gemini(user_question)
                st.markdown("**Answer:**")
                st.write(answer)
        else:
            st.warning("Please enter a question.")