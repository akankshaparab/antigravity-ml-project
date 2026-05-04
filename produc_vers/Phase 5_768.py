import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Configuration & Styling ---
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'

def run_phase_5_evaluation():
    print("--- Phase 5: Deep Evaluation Metrics (768 Dimensions) ---")
    
    # 1. Load Data (Using 768D)
    input_file = 'spider_768_embeddings.npz'
    print(f"Loading embeddings from {input_file}...")
    data = np.load(input_file)
    X_full = data['X']
    y = data['y']
    
    # Split
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train the Optimized Model
    pca = PCA(n_components=0.90).fit(X_train_full)
    X_train = pca.transform(X_train_full)
    X_test = pca.transform(X_test_full)

    print(f"[Step 1] Training Optimized SVM (PCA={pca.n_components_} for 90% Var)...")
    clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 3. Comprehensive Classification Report
    print("\n[Step 2] Generating Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.iloc[:-3].round(4))

    # 4. === QUERY ROUTING LAYER PoC ===
    print("\n--- Testing Routing Layer (768D Model) ---")
    routing_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    sample_queries = [
        "Select all the stars in the constellation Cygnus", 
        "Show average age and department for employees joined before 2010 with salaries over 50000"
    ]
    
    for q in sample_queries:
        vec = routing_model.encode([q], normalize_embeddings=True)
        low_dim_vec = pca.transform(vec)
        prediction = clf.predict(low_dim_vec)[0]
        target = "GEMINI FLASH" if prediction in ["Easy", "Medium"] else "GEMINI PRO"
        print(f"Query: '{q}'\nDecision: {prediction} -> Route to {target}\n")

if __name__ == "__main__":
    run_phase_5_evaluation()
