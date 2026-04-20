import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd

# --- Configuration & Styling ---
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
primary_color = '#6366f1' # Indigo
secondary_color = '#ec4899' # Pink

def run_phase_5_evaluation():
    print("--- Phase 5: Deep Evaluation Metrics ---")
    
    # 1. Load Data
    input_file = 'spider_reduced_embeddings.npz'
    data = np.load(input_file)
    X_full = data['X']
    y = data['y']
    
    # Split (same seed as Phase 4 for consistency)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train the Optimized Model (using 50 dimensions as found in Phase 4)
    n_comp = 50
    pca = PCA(n_components=n_comp).fit(X_train_full)
    X_train = pca.transform(X_train_full)
    X_test = pca.transform(X_test_full)

    print(f"[Step 1] Training Optimized SVM (PCA={n_comp}, Kernel=RBF)...")
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 3. Comprehensive Classification Report
    print("\n[Step 2] Generating Classification Report (Precision, Recall, F1):")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.round(4))

    # 4. Confusion Matrix Visualization
    print("\n[Step 3] Creating Confusion Matrix...")
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Normalize by row (to show recall-like percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized Scale'})
    
    plt.title("Confusion Matrix: Difficulty Classification", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('phase5_confusion_matrix.png', dpi=300)
    print("- Confusion Matrix saved as 'phase5_confusion_matrix.png'")

    # 5. Focused Analysis: catching "Hard" Queries
    print("\n[Step 4] Analyzing 'Hard' & 'Extra Hard' Catch Rate (Recall)...")
    hard_recall = report['Hard']['recall']
    extra_hard_recall = report['Extra Hard']['recall']
    
    print(f"  - Recall for 'Hard' queries: {hard_recall:.2%}")
    print(f"  - Recall for 'Extra Hard' queries: {extra_hard_recall:.2%}")

    # Visualizing Recall per Class
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    class_names = labels
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, [report[c]['precision'] for c in class_names], width, label='Precision', color='#10b981')
    ax.bar(x, [report[c]['recall'] for c in class_names], width, label='Recall', color='#6366f1')
    ax.bar(x + width, [report[c]['f1-score'] for c in class_names], width, label='F1-Score', color='#f59e0b')

    ax.set_title("Metric Comparison Across Difficulty Levels", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('phase5_metric_comparison.png', dpi=300)
    print("- Metric Comparison plot saved as 'phase5_metric_comparison.png'")

    print("\nPhase 5 Conclusion:")
    print("Look at the Confusion Matrix to see where the model 'mixes up' categories.")
    print("The goal in Phase 6 might be to improve the Recall for 'Hard' items if they are being missed.")

if __name__ == "__main__":
    run_phase_5_evaluation()
