import streamlit as st
import pandas as pd
pd.set_option("styler.render.max_elements", 10000000)
import joblib
import os
import io

st.set_page_config(
    page_title="NSL-KDD Intrusion Dashboard",
    page_icon="🛡️",
    layout="wide"
)

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", 
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate"
]

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'intrusion_detection_model.pkl')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, 'intrusion_detection_model.pkl')
        if not os.path.exists(model_path):
            return None
            
    return joblib.load(model_path)

def main():
    st.title("🛡️ NSL-KDD Network Intrusion Dashboard")
    st.markdown("""
    Upload a batch of network traffic logs (CSV/TXT format) to analyze them for malicious anomalies.
    The model is specifically trained on the **NSL-KDD dataset** schema.
    """)
    
    model = load_model()
    
    if model is None:
        st.error("Model array not found! Ensure the CI pipeline completed and saved to `models/`.")
        return
        
    uploaded_file = st.file_uploader("Upload Network Traffic logs (.csv or .txt)", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            try:
                # NSL-KDD files often have no headers, so we enforce our known columns if the width is 43
                # For safety, we try to load it normally. If it fails due to column mismatch, we fallback
                df_raw = pd.read_csv(uploaded_file, header=None)
                
                if df_raw.shape[1] >= 41:
                    # Assume the first 41 are our features
                    df_features = df_raw.iloc[:, :41].copy()
                    df_features.columns = FEATURE_COLUMNS
                else:
                    st.error("Uploaded file does not have enough columns. Expected at least 41 features.")
                    return
                
                # Inference
                predictions = model.predict(df_features)
                probabilities = model.predict_proba(df_features)[:, 1]
                
                # Append Results
                df_results = df_features.copy()
                df_results['Threat_Detected'] = ["🚨 Intrusion" if p == 1 else "✅ Normal" for p in predictions]
                df_results['Anomaly_Confidence'] = [f"{prob:.2%}" for prob in probabilities]
                
                # Reposition results to the front
                cols = ['Threat_Detected', 'Anomaly_Confidence'] + [c for c in df_results.columns if c not in ['Threat_Detected', 'Anomaly_Confidence']]
                df_results = df_results[cols]
                
                st.success(f"Log analysis complete! Found {sum(predictions)} threats out of {len(predictions)} connections.")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Connections Analyzed", len(predictions))
                col2.metric("Intrusions Detected", sum(predictions), delta_color="inverse")
                col3.metric("Normal Traffic", len(predictions) - sum(predictions))
                
                st.subheader("Analysis Breakdown (Preview)")
                
                filter_choice = st.radio("Filter Traffic:", ["All", "Only Intrusions", "Only Normal"], horizontal=True)
                
                if filter_choice == "Only Intrusions":
                    display_df = df_results[df_results['Threat_Detected'].str.contains("Intrusion")]
                elif filter_choice == "Only Normal":
                    display_df = df_results[df_results['Threat_Detected'].str.contains("Normal")]
                else:
                    display_df = df_results
                    
                st.caption(f"Showing top 1000 out of {len(display_df)} matching records:")
                
                st.dataframe(
                    display_df.head(1000).style.map(lambda x: 'background-color: #ffcccc' if "🚨" in str(x) else '', subset=['Threat_Detected']),
                    use_container_width=True
                )
                
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Analysis Results (CSV)",
                    data=csv,
                    file_name='network_intrusion_analysis.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Failed to process dataset: {e}")
                
if __name__ == '__main__':
    main()
