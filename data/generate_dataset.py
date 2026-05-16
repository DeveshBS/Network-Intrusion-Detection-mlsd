import pandas as pd
import numpy as np
import os

def generate_network_traffic_data(num_samples=5000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 90% normal traffic, 10% intrusion
    labels = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
    
    data = []
    for label in labels:
        if label == 0:  # Normal Traffic
            duration = np.random.exponential(scale=1.5)
            protocol_type = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.8, 0.15, 0.05])
            src_bytes = int(np.random.normal(loc=1500, scale=300))
            dst_bytes = int(np.random.normal(loc=3000, scale=800))
            count = int(np.random.poisson(lam=5))
            srv_count = int(np.random.poisson(lam=5))
            same_srv_rate = np.random.uniform(0.9, 1.0)
            diff_srv_rate = np.random.uniform(0.0, 0.1)
        else:  # Intrusion Anomaly
            duration = np.random.exponential(scale=10.0) # Longer connections
            protocol_type = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.4, 0.4, 0.2])
            src_bytes = int(np.random.normal(loc=8000, scale=2000)) # Larger payloads
            dst_bytes = int(np.random.normal(loc=500, scale=200))
            count = int(np.random.poisson(lam=50)) # Higher frequency (e.g. DoS)
            srv_count = int(np.random.poisson(lam=2))
            same_srv_rate = np.random.uniform(0.0, 0.3) # More varied services
            diff_srv_rate = np.random.uniform(0.8, 1.0)
            
        # Ensure positive byte values
        src_bytes = max(0, src_bytes)
        dst_bytes = max(0, dst_bytes)
        
        data.append([
            duration, protocol_type, src_bytes, dst_bytes, 
            count, srv_count, same_srv_rate, diff_srv_rate, label
        ])
        
    cols = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 
            'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate', 'class']
    
    df = pd.DataFrame(data, columns=cols)
    return df

if __name__ == "__main__":
    print("Generating synthetic network intrusion dataset...")
    df = generate_network_traffic_data(5000)
    
    # Create data directory if not exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_traffic.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Class distribution:\n{df['class'].value_counts(normalize=True)}")
