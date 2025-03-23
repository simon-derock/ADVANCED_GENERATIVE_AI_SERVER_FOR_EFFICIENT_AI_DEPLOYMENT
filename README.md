# Portable AI Server with Ollama - RAG Enhanced

***Developed by: PHILIP SIMON DEROCK P***

## Overview
This project transforms an **old laptop** into a **dual-boot AI server** running Ubuntu Server and Windows. The server is fully **wireless**, allowing for **AI interaction using Ollama** and implementing **Database + Retrieval-Augmented Generation (RAG)** for enhanced AI responses.

## Features
- **Dual-Boot Setup**: Ubuntu Server + Windows 10
- **Wireless AI Server**: No Ethernet needed, runs on Wi-Fi/mobile hotspot
- **Ollama AI**: Chat-based AI with multiple models
- **MySQL Integration**: Stores chat history and responses
- **RAG-Enhanced AI**: Uses embeddings and similarity search for better results
- **Streamlit Web UI**: Dark-themed interface with model selection
- **Secure SSH Access**: Remote management capability
- **Mobile Hotspot Connectivity**: No need for a static network

---

## 1️⃣ Setting Up the Dual-Boot Server

### Hardware Requirements
To successfully set up the AI server, ensure the following hardware:
- **Old Laptop** (Recommended: Intel i5 or better, 8GB RAM, SSD for optimal performance)
- **USB Drive (8GB+)** for Ubuntu installation
- **Stable Wi-Fi Connection**
- **External Storage (Optional)** for additional backup and data storage

### Step-by-Step Guide to Installing Ubuntu Server alongside Windows
1. **Create a Bootable USB Drive**:
   - Download the **Ubuntu Server 22.04.2 ISO** from [Ubuntu's official website](https://ubuntu.com/download/server).
   - Use **Rufus** (Windows) or **Balena Etcher** (Mac/Linux) to create a bootable USB drive.
2. **Partition the Disk for Dual-Boot**:
   - Open **Disk Management** in Windows (`diskmgmt.msc`).
   - Shrink an existing partition to free up at least **40GB** for Ubuntu.
   - Leave the space **unallocated**.
3. **Install Ubuntu Server**:
   - Boot from the USB drive and follow the installation prompts.
   - Select **Manual Partitioning** and assign the free space:
     - `/` (root) - 20GB
     - `/home` - 15GB
     - `swap` - 4GB
   - Install the **OpenSSH server** during installation for remote access.
4. **Set Up GRUB Bootloader**:
   - Ubuntu will automatically detect Windows and configure **GRUB** to enable boot selection.
   - Verify by restarting and ensuring you can select **Ubuntu Server** or **Windows** at boot.

---

## 2️⃣ Configuring Wireless AI Server

### Connecting Ubuntu Server to Wi-Fi
By default, Ubuntu Server does not have a graphical network manager. Configure Wi-Fi manually:
1. **Find the Wireless Interface**:
   ```sh
   iwconfig
   ```
2. **Modify the Netplan Configuration**:
   - Edit the configuration file:
     ```sh
     sudo nano /etc/netplan/50-cloud-init.yaml
     ```
   - Add the following content:
     ```yaml
     network:
       version: 2
       renderer: networkd
       wifis:
         wlan0:
           dhcp4: true
           access-points:
             "Your_WiFi_SSID":
               password: "Your_WiFi_Password"
     ```
   - Apply changes:
     ```sh
     sudo netplan apply
     ```
3. **Verify Connection**:
   ```sh
   ip a
   ```
   - Look for an assigned IP under `wlan0`.

### Setting Up SSH for Remote Access
SSH enables managing the server without physical access:
1. **Enable SSH**:
   ```sh
   sudo systemctl enable ssh
   sudo systemctl start ssh
   ```
2. **Find the Server's IP Address**:
   ```sh
   hostname -I
   ```
3. **Access the Server Remotely**:
   ```sh
   ssh user@server-ip
   ```

---

## 3️⃣ Installing Python and Required Libraries

### Installing Python
1. **Update the package list and install Python**:
   ```sh
   sudo apt update && sudo apt install python3 python3-pip -y
   ```

2. **Verify Python installation**:
   ```sh
   python3 --version
   ```

### Installing Required Libraries
Run the following command to install all required dependencies:
```sh
python3 -m pip install streamlit
python3 -m pip install mysql-connector-python
python3 -m pip install numpy
python3 -m pip install langchain_ollama
python3 -m pip install langchain_core
python3 -m pip install scikit-learn

```

---

## 4️⃣ Deploying Ollama AI on the Server

### Installing Ollama & AI Models
1. **Install Ollama**:
   ```sh
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
2. **Pull AI Models**:
   ```sh
   ollama pull deepseek-r1:1.5b
   ollama pull tinyllama:latest
   ollama pull lucianotonet/llamaclaude:latest
   ollama pull incept5/IIama3-I-claude:latest
   ollama pull granite3-dense:latest
   ollama pull llama2-uncensored:latest
   ollama pull deepseek-rl:1.5b
   ```

### Running Ollama AI Server
To make Ollama accessible over the network:
```sh
export OLLAMA_HOST=0.0.0.0
ollama serve
```

---

## 5️⃣ Implementing Database & RAG

### MySQL Database Schema
```sql
CREATE DATABASE cld;
USE cld;

CREATE TABLE conversations (
    session_id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE c_tbl (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36),
    message_order INT,
    prompt TEXT,
    response TEXT,
    embedding TEXT,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
);
```

---

## 6️⃣ Running the AI Chat Interface
### Full `app.py` Code
```python
import streamlit as st
import mysql.connector
import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import time

# Debugging Log
def log_debug(message):
    st.sidebar.text(f"DEBUG: {message}")  # Display in sidebar
    print(f"DEBUG: {message}")  # Print to console

# MySQL Database Connection
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="cld"
        )
    except mysql.connector.Error as err:
        st.sidebar.error(f"Database Error: {err}")
        return None

# Initialize database tables
def initialize_db():
    db = connect_db()
    if db is None:
        return
    
    try:
        cursor = db.cursor()
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id VARCHAR(36) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Updated chat history table with conversation context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS c_tbl (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36),
                message_order INT,
                prompt TEXT,
                response TEXT,
                embedding TEXT,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES conversations(session_id)
            )
        """)
        
        # Documents table for RAG
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255),
                content TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        db.commit()
        cursor.close()
        db.close()
        log_debug("Database initialized successfully.")
    except mysql.connector.Error as err:
        st.sidebar.error(f"Database Initialization Error: {err}")

# Get or create session ID
def get_session_id():
    if "session_id" not in st.session_state:
        # Create a new session ID
        st.session_state.session_id = str(uuid.uuid4())
        # Save to database
        save_conversation(st.session_state.session_id)
    return st.session_state.session_id

# Save conversation record
def save_conversation(session_id):
    db = connect_db()
    if db is None:
        return
    
    try:
        cursor = db.cursor()
        cursor.execute("INSERT INTO conversations (session_id) VALUES (%s)", (session_id,))
        db.commit()
        cursor.close()
        db.close()
        log_debug(f"Conversation {session_id} saved successfully.")
    except mysql.connector.Error as err:
        st.sidebar.error(f"Database Error: {err}")

# Save chat history to MySQL with context
def save_chat_history(prompt, response, embedding, context=""):
    session_id = get_session_id()
    db = connect_db()
    if db is None:
        return
    
    try:
        cursor = db.cursor()
        # Get the next message order for this session
        cursor.execute("SELECT COALESCE(MAX(message_order), 0) FROM c_tbl WHERE session_id = %s", (session_id,))
        message_order = cursor.fetchone()[0] + 1
        
        cursor.execute("""
            INSERT INTO c_tbl 
            (session_id, message_order, prompt, response, embedding, context) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (session_id, message_order, prompt, response, str(embedding) if embedding else "[]", context))
        
        db.commit()
        cursor.close()
        db.close()
        log_debug("Chat history saved successfully.")
    except mysql.connector.Error as err:
        st.sidebar.error(f"Database Error: {err}")

# Add a document to the RAG system
def add_document(title, content, embedding):
    db = connect_db()
    if db is None:
        return
    
    try:
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO documents (title, content, embedding) 
            VALUES (%s, %s, %s)
        """, (title, content, str(embedding) if embedding else "[]"))
        
        db.commit()
        cursor.close()
        db.close()
        log_debug(f"Document '{title}' added successfully.")
    except mysql.connector.Error as err:
        st.sidebar.error(f"Database Error: {err}")

# Load External CSS
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styling.")

# Function to get embedding vector
def get_embedding(text):
    if embedding_engine:
        try:
            return embedding_engine.embed_query(text)
        except Exception as e:
            st.sidebar.error(f"Embedding error: {e}")
    return None

# Generate conversation context
def generate_context(user_query):
    # Get recent conversation history
    session_id = get_session_id()
    db = connect_db()
    if db is None:
        return ""
    
    try:
        cursor = db.cursor()
        # Get last 3 conversation turns from the current session
        cursor.execute("""
            SELECT prompt, response 
            FROM c_tbl 
            WHERE session_id = %s 
            ORDER BY message_order DESC 
            LIMIT 3
        """, (session_id,))
        
        records = cursor.fetchall()
        cursor.close()
        db.close()
        
        # Create context string
        context = ""
        for prompt, response in reversed(records):  # Show oldest first
            context += f"User: {prompt}\nAI: {response}\n\n"
        
        return context.strip()
    except mysql.connector.Error as err:
        st.sidebar.error(f"Database Error: {err}")
        return ""

# Fetch stored embeddings from the database with context awareness
def fetch_embeddings():
    session_id = get_session_id()
    db = connect_db()
    if db is None:
        return [], np.array([])
    
    cursor = db.cursor()
    # Get all entries from current session
    cursor.execute("""
        SELECT prompt, response, embedding, context 
        FROM c_tbl 
        WHERE session_id = %s 
        ORDER BY message_order
    """, (session_id,))
    
    records = cursor.fetchall()
    cursor.close()
    db.close()
    
    prompts, responses, embeddings, contexts = [], [], [], []
    for prompt, response, stored_embedding, context in records:
        try:
            embedding_array = np.array(literal_eval(stored_embedding))
            if embedding_array.size > 0:
                embeddings.append(embedding_array)
                prompts.append(prompt)
                responses.append(response)
                contexts.append(context)
        except:
            pass
    
    return list(zip(prompts, responses, contexts)), np.array(embeddings)

# Search for similar documents in the RAG system
def search_rag_documents(query_embedding, top_k=3, threshold=0.7):
    db = connect_db()
    if db is None:
        return []
    
    cursor = db.cursor()
    cursor.execute("SELECT id, title, content, embedding FROM documents")
    records = cursor.fetchall()
    cursor.close()
    db.close()
    
    doc_ids, titles, contents, embeddings = [], [], [], []
    for doc_id, title, content, stored_embedding in records:
        try:
            embedding_array = np.array(literal_eval(stored_embedding))
            if embedding_array.size > 0:
                doc_ids.append(doc_id)
                titles.append(title)
                contents.append(content)
                embeddings.append(embedding_array)
        except:
            pass
    
    if not embeddings:
        return []
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top-k documents above threshold
    results = []
    for i in range(len(similarities)):
        if similarities[i] >= threshold:
            results.append((doc_ids[i], titles[i], contents[i], similarities[i]))
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x[3], reverse=True)
    
    # Return top-k results
    return results[:top_k]

# Similarity Search Function with context awareness
def find_similar_response(user_query, user_embedding, threshold=0.85):
    conversation_context = generate_context(user_query)
    log_debug(f"Current context: {conversation_context}")
    
    # First check for similar responses in the same conversation
    data, embeddings = fetch_embeddings()
    if embeddings.size > 0:
        similarities = cosine_similarity([user_embedding], embeddings)[0]
        best_match_index = np.argmax(similarities)
        
        if similarities[best_match_index] >= threshold:
            # Check if there's sufficient context match
            matched_prompt, matched_response, matched_context = data[best_match_index]
            log_debug(f"Best match: {matched_prompt} (score: {similarities[best_match_index]:.2f})")
            
            # If this is a fresh session or the query is a follow-up, consider context
            if not conversation_context or matched_context in conversation_context:
                log_debug("Context match found, using cached response.")
                return matched_response
            else:
                log_debug("Context mismatch, ignoring match.")
    
    # If no suitable match in conversation history, search RAG documents
    rag_results = search_rag_documents(user_embedding)
    if rag_results:
        # Use RAG results to augment the prompt
        context_docs = "\n\n".join([f"Document: {title}\n{content}" for _, title, content, _ in rag_results])
        log_debug(f"Found {len(rag_results)} relevant documents for RAG.")
        return None, context_docs
    
    log_debug("No similar response or relevant documents found.")
    return None, ""

# Streamlit UI
st.title("Portable AI Server With Ollama - RAG Enhanced")
st.caption("Private AI Chat Interface with RAG")

# Initialize database on first run
if "db_initialized" not in st.session_state:
    initialize_db()
    st.session_state.db_initialized = True

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "llama2-uncensored:latest", "nomic-embed-text:latest"],
        index=0
    )
    
    # RAG Controls
    # st.header("📚 RAG System")
    # with st.expander("Add Document"):
    #     doc_title = st.text_input("Document Title")
    #     doc_content = st.text_area("Document Content")
    #     if st.button("Add to Knowledge Base"):
    #         if doc_title and doc_content:
    #             doc_embedding = get_embedding(doc_content)
    #             add_document(doc_title, doc_content, doc_embedding)
    #             st.success(f"Document '{doc_title}' added successfully!")
    #         else:
    #             st.warning("Title and content are required.")

# Initialize Embedding Engine
load_css("style.css")

# Initialize LLM and Embedding Engines
llm_engine = None
embedding_engine = None

try:
    llm_engine = ChatOllama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0.3,
        streaming=True
    )
    log_debug("LLM Engine initialized successfully.")
except Exception as e:
    st.sidebar.error(f"LLM Engine Error: {e}")

try:
    embedding_engine = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )
    log_debug("Embedding Engine initialized successfully.")
except Exception as e:
    st.sidebar.error(f"Embedding model error: {e}")

# System prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert AI coding assistant. If you see `{{i}}` or `{{stack_size}}`, treat them as placeholders.
    
    If relevant information is provided in the context below, use it to inform your response:
    {rag_context}
    """
)

# Maintain Chat History
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! How can I help you..."}]

# Display chat history
for message in st.session_state.message_log:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Box
user_query = st.chat_input("Message Private AI")

# AI Response Generator
def generate_ai_response(prompt_chain, rag_context=""):
    if not llm_engine:
        st.error("LLM Engine is not available.")
        return ""

    response_text = ""
    try:
        for chunk in (prompt_chain | llm_engine | StrOutputParser()).stream({"rag_context": rag_context}):
            yield chunk
            response_text += chunk
    except Exception as e:
        st.error(f"AI Response Error: {e}")
    
    return response_text

# Build Prompt Chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        content_fixed = msg["content"].replace("{", "{{").replace("}", "}}")
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(content_fixed))
        else:
            prompt_sequence.append(AIMessagePromptTemplate.from_template(content_fixed))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# New Session Button
if st.sidebar.button("Start New Conversation"):
    # Reset session
    if "session_id" in st.session_state:
        del st.session_state.session_id
    st.session_state.message_log = [{"role": "ai", "content": "Hi! How can I help you..."}]
    st.rerun()

# Process User Query
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    user_embedding = get_embedding(user_query)
    
    # Get current conversation context
    conversation_context = generate_context(user_query)
    
    # Check for similar past queries
    ai_response_text = None
    rag_context = ""
    
    if user_embedding is not None:
        ai_response_text, rag_context = find_similar_response(user_query, user_embedding)
    
    if not ai_response_text:
        with st.chat_message("ai"):
            ai_response_placeholder = st.empty()
            with st.spinner("🧠 Processing..."):
                ai_response_text = "".join(generate_ai_response(build_prompt_chain(), rag_context))
            ai_response_placeholder.markdown(ai_response_text)
    else:
        with st.chat_message("ai"):
            st.markdown(ai_response_text)
    
    # Save history & update UI
    st.session_state.message_log.append({"role": "ai", "content": ai_response_text})
    save_chat_history(user_query, ai_response_text, user_embedding, conversation_context)
    st.rerun()
```

### Full `style.css` Code
```css
/* ===== Fully Black Background (Chromatic Black) ===== */
body, .stApp, .sidebar .sidebar-content {
    background-color: #0a0a0a !important; /* Chromatic Black */
    color: #00ff00 !important; /* Neon Green Text */
}

/* ===== Sidebar Fully Black ===== */
.sidebar .sidebar-content {
    background-color: #0a0a0a !important;
    border-right: none !important; /* Removes outline */
}

/* ===== Normal Text (Paragraphs, Labels, General Content) in Green ===== */
p, span, div, label, 
.stTextInput textarea, 
.stSelectbox div[data-baseweb="select"], 
.stButton>button, 
code, pre {
    color: #00ff00 !important; /* Neon Green */
}

/* ===== Subtopics (h2, h3, h4, h5, h6) in Red (Fixed Glitch) ===== */
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4, 
.stMarkdown h5, .stMarkdown h6,
h2, h3, h4, h5, h6 {
    color: #ff0000 !important; /* Forcefully Red */
    font-weight: bold !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    background-color: transparent !important; /* Prevent overlays */
}

/* ===== Buttons & Inputs Fully Black with Green Text ===== */
.stButton>button, 
.stTextInput textarea, 
.stSelectbox div[data-baseweb="select"], 
.stSelectbox option, 
div[role="listbox"] div {
    background-color: #0a0a0a !important; /* Full Black */
    color: #00ff00 !important; /* Green Text */
    border: none !important;
}

/* ===== Chat Messages Fully Black with Green Text ===== */
.stChatMessage {
    background-color: #0a0a0a !important;
    color: #00ff00 !important;
}

/* ===== Code Blocks (Green Text on Black Background) ===== */
code, pre {
    background-color: #0a0a0a !important; /* Black */
    color: #00ff00 !important; /* Green */
    border-radius: 5px;
    padding: 10px;
}

/* ===== Remove All Unwanted Borders and Outlines ===== */
* {
    outline: none !important;
    box-shadow: none !important;
}

```

### Running the App
```sh
streamlit run four.py
```

---

## Conclusion
This project successfully transforms an **old laptop** into a **fully wireless AI server** that supports **retrieval-augmented generation (RAG)** and provides a **database-backed chatbot** using **MySQL, Ollama AI, and Streamlit**.

### ⭐ Star this repo if you found it useful!
