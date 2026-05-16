import kagglehub
import os
import shutil

def fetch_data():
    print("Downloading NSL-KDD dataset from Kaggle...")
    # Fetch the dataset using kagglehub
    path = kagglehub.dataset_download("hassan06/nslkdd")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We specifically need the Train and Test files
    target_files = ['KDDTrain+.txt', 'KDDTest+.txt']
    
    # The dataset might be nested in a subfolder or directly in the path depending on Kaggle's extraction
    # We will search recursively inside the downloaded path
    found_files = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file in target_files:
                src = os.path.join(root, file)
                dst = os.path.join(current_dir, file)
                shutil.copy2(src, dst)
                print(f" Successfully copied {file} to {current_dir}")
                found_files += 1
                
    if found_files < len(target_files):
        print(" Warning: Some required files were not found in the downloaded package.")

if __name__ == "__main__":
    fetch_data()
