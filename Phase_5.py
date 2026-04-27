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
    print("--- Phase 5: Deep Evaluation Metrics & PoC ---")
    
    # 1. Load Data (Using ORIGINAL 384D to match raw model outputs)
    input_file = 'spider_final_embeddings.npz'
    print(f"Loading embeddings from {input_file}...")
    data = np.load(input_file)
    X_full = data['X']
    y = data['y']
    
    # Split
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train the Optimized Model
    # Retain 90% variance to capture structural complexity
    pca = PCA(n_components=0.90).fit(X_train_full)
    X_train = pca.transform(X_train_full)
    X_test = pca.transform(X_test_full)

    n_actual = pca.n_components_
    print(f"[Step 1] Training Optimized SVM (PCA={n_actual} for 90% Var, Kernel=RBF)...")
    # Added class_weight='balanced' to handle 'Extra Hard' minority class
    clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 3. Comprehensive Classification Report
    print("\n[Step 2] Generating Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.iloc[:-3].round(4)) # Exclude global metrics for clean view

    # 4. Confusion Matrix Visualization
    print("\n[Step 3] Creating Confusion Matrix...")
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix: Difficulty Classification", fontsize=15, fontweight='bold', pad=20)
    plt.savefig('phase5_confusion_matrix.png', dpi=300)
    print("- Confusion Matrix saved.")

    # 5. Identifying 'Poorly Represented' Query Types (Research Objective)
    print("\n[Analysis] Characterizing 'Poorly Represented' Types:")
    low_recall_class = df_report.iloc[:-3]['recall'].idxmin()
    recall_value = df_report.loc[low_recall_class, 'recall']
    print(f"  - Worst Performing Type: '{low_recall_class}'")
    print(f"  - Identifying Gap: Only {recall_value:.1%} recall. This type needs more diverse training samples.")

    # Visualization: Metric Comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [report[c]['precision'] for c in labels], 0.2, label='Precision', color='#10b981')
    ax.bar(x, [report[c]['recall'] for c in labels], 0.2, label='Recall', color='#6366f1')
    ax.bar(x + 0.2, [report[c]['f1-score'] for c in labels], 0.2, label='F1-Score', color='#f59e0b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('phase5_metric_comparison.png', dpi=300)
    print("- Metric Comparison plot saved.")

    # 6. === QUERY ROUTING LAYER PoC (Final Goal) ===
    print("\n--- Testing Routing Layer (Practical Application) ---")
    routing_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    sample_queries = [
        "Select all the stars in the constellation Cygnus", 
        "Show average age and department for employees joined before 2010 with salaries over 50000"
    ]
    
    for q in sample_queries:
        vec = routing_model.encode([q], normalize_embeddings=True)
        # Now matches: 384D input -> PCA fit on 384D
        low_dim_vec = pca.transform(vec)
        prediction = clf.predict(low_dim_vec)[0]
        target = "GEMINI FLASH" if prediction in ["Easy", "Medium"] else "GEMINI PRO"
        print(f"Query: '{q}'\nDecision: {prediction} -> Route to {target}\n")

    print("Phase 5 Complete. All research objectives and stretch goals implemented.")

if __name__ == "__main__":
    run_phase_5_evaluation()
