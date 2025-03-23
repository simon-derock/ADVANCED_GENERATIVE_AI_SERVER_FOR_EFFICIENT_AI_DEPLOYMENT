# Portable AI Server with Ollama - RAG Enhanced

**Developed by: PHILIP SIMON DEROCK P**

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

## 3️⃣ Deploying Ollama AI on the Server

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
   ollama pull incept5/IIama3.I—claude:latest
   ollama pull granite3—dense:latest
   ollama pull llama2—uncensored:latest
   ollama pull deepseek—rl:1.5b
   ```

### Running Ollama AI Server
To make Ollama accessible over the network:
```sh
OLLAMA_HOST=0.0.0.0 ollama serve
```

---

## 4️⃣ Implementing Database & RAG

### Installing MySQL and Configuring the Database
1. **Install MySQL**:
   ```sh
   sudo apt install mysql-server -y
   ```
2. **Secure MySQL Installation**:
   ```sh
   sudo mysql_secure_installation
   ```
3. **Create a Database for Chat Storage**:
   ```sql
   CREATE DATABASE ai_chat_db;
   ```
4. **Create Tables**:
   ```sql
   USE ai_chat_db;
   CREATE TABLE c_tbl (
       id INT AUTO_INCREMENT PRIMARY KEY,
       user_query TEXT,
       ai_response TEXT,
       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   CREATE TABLE conversations (
       id INT AUTO_INCREMENT PRIMARY KEY,
       session_id VARCHAR(255),
       user_message TEXT,
       bot_response TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

---

## 5️⃣ Hosting the AI Chat Interface
### Running Streamlit-based Web UI
1. **Create `app.py`**:
   ```python
   import streamlit as st
   import mysql.connector

   st.title("Portable AI Server - RAG Enhanced")
   user_input = st.text_input("Ask me anything:")

   if st.button("Submit"):
       conn = mysql.connector.connect(host='localhost', user='root', password='yourpassword', database='ai_chat_db')
       cursor = conn.cursor()
       cursor.execute("SELECT ai_response FROM c_tbl WHERE user_query=%s", (user_input,))
       result = cursor.fetchone()
       st.write(result[0] if result else "No data found.")
   ```
2. **Run the App**:
   ```sh
   streamlit run app.py
   ```

---

## Conclusion
This project successfully transforms an **old laptop** into a **fully wireless AI server** that supports **retrieval-augmented generation (RAG)** and provides a **database-backed chatbot** using **MySQL, Ollama AI, and Streamlit**.

### ⭐ Star this repo if you found it useful!

