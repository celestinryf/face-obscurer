import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Privacy Engine Dashboard", page_icon="🛡️", layout="wide")

# Custom CSS for a dark, professional hackathon look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Privacy Engine: Live Protection Dashboard")
st.caption("Right to be Forgotten - Real-time Face Recognition & Censorship")
st.markdown("---")

# Setup Sidebar and Columns
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("⚙️ System Control")
    
    # 1. Connection & Health Check
    try:
        health_res = requests.get("http://127.0.0.1:8000/health").json()
        st.success(f"Connected to Backend (v{health_res['version']})")
        
        # Display Quick Stats
        s1, s2 = st.columns(2)
        s1.metric("Enrolled Users", health_res['total_enrolled_users'])
        s2.metric("System Status", "Healthy")
    except:
        st.error("❌ Backend Offline: Ensure uvicorn is running.")

    st.markdown("---")
    
    # 2. Database Management
    st.subheader("Vault Management")
    if st.button("🔄 Sync Known Faces Directory"):
        with st.spinner("Processing local images..."):
            sync_res = requests.post("http://127.0.0.1:8000/sync-known-faces").json()
            st.info(f"Synced: {sync_res['synced']} | Skipped: {sync_res['skipped']}")
            st.rerun()

    # 3. List Enrolled Users
    st.subheader("👤 Protected Identities")
    try:
        users_res = requests.get("http://127.0.0.1:8000/users").json()
        if users_res['users']:
            df = pd.DataFrame(users_res['users'])
            st.dataframe(df[['alias', 'user_id', 'created_at']], hide_index=True, use_container_width=True)
        else:
            st.write("No users currently enrolled.")
    except:
        st.write("Could not load user list.")

with col2:
    st.header("📹 Live Protected Stream")
    
    # This calls the /video_feed endpoint you just added to main.py
    # Streamlit treats the URL as a constant image update
    st.image("http://127.0.0.1:8000/video_feed", 
             caption="Real-time Stream: Faces of enrolled users are automatically blacked out.",
             use_container_width=True)
    
    st.markdown("""
    ### 🛡️ How it works
    - **Detection:** dlib scans for 128-dimensional biometric facial embeddings.
    - **Encryption:** Only face vectors are stored in the database; original images are discarded.
    - **Zero-Latency Privacy:** The system recognizes and obscures 'Opt-Out' users before the frame is rendered.
    """)