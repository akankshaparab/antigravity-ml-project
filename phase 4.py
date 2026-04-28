import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas as pd
import time

# --- Configuration & Styling ---
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'

def run_phase_4_optimization():
    print("--- Phase 4: Optimization and Prediction ---")
    
    # 1. Load Data (Using the original 384D to allow varying components)
    input_file = 'spider_reduced_embeddings.npz'
    data = np.load(input_file)
    X_full = data['X']
    y = data['y']
    
    # Split into training and testing sets (80/20)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- TASK 1 & 2: Kernel Comparison (using 50 components as baseline) ---
    print("\n[Step 1-2] Testing Kernel Functions (n_components=50)...")
    pca_50 = PCA(n_components=50).fit(X_train_full)
    X_train_50 = pca_50.transform(X_train_full)
    X_test_50 = pca_50.transform(X_test_full)

    kernels = ['linear', 'rbf', 'poly']
    kernel_results = {}

    for kernel in kernels:
        start = time.time()
        clf = SVC(kernel=kernel, C=1.0)
        clf.fit(X_train_50, y_train)
        y_pred = clf.predict(X_test_50)
        acc = accuracy_score(y_test, y_pred)
        duration = time.time() - start
        kernel_results[kernel] = acc
        print(f"  - {kernel.upper()} Kernel Accuracy: {acc:.4f} (Time: {duration:.2f}s)")

    # --- TASK 2.5: Visualization: Kernel Comparison Bar Chart ---
    plt.figure(figsize=(10, 6))
    kernel_names = [k.upper() for k in kernel_results.keys()]
    accuracies = list(kernel_results.values())
    
    # Create bar plot with a premium color palette
    sns.barplot(x=kernel_names, y=accuracies, palette='magma', hue=kernel_names, legend=False)
    
    # Styling and Labels
    plt.title('Comparison of SVM Kernel Accuracies (50 Components)', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Kernel Function Type', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0, 1.1)  # Leave space for labels
    
    # Add accuracy values on top of the bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase4_kernel_comparison.png', dpi=300)
    print("\n[SUCCESS] Kernel comparison plot saved as 'phase4_kernel_comparison.png'")


    # --- TASK 3: Sensitivity Analysis ---
    print("\n[Step 3] Performing Sensitivity Analysis (Varying Dimensions)...")
    dimensions = [5, 10, 20, 30, 50, 75, 100, 150]
    sensitivity_metrics = []

    for n in dimensions:
        print(f"  - Testing {n} components...", end="\r")
        # Apply PCA for specific dimension count
        pca = PCA(n_components=n).fit(X_train_full)
        X_train_n = pca.transform(X_train_full)
        X_test_n = pca.transform(X_test_full)
        
        # Train RBF SVM (usually the best performer)
        clf = SVC(kernel='rbf')
        clf.fit(X_train_n, y_train)
        y_pred = clf.predict(X_test_n)
        
        # Record metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        
        sensitivity_metrics.append({
            'Dimensions': n,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec
        })

    df_metrics = pd.DataFrame(sensitivity_metrics)
    print("\nSensitivity Metrics Table:")
    print(df_metrics)

    # --- TASK 4: Find the Elbow Point & Visualize ---
    print("\n[Step 4] Plotting Results to find the Elbow Point...")
    plt.figure(figsize=(12, 7))
    
    # Plot Accuracy, Precision, and Recall
    plt.plot(df_metrics['Dimensions'], df_metrics['Accuracy'], marker='o', lw=3, label='Accuracy', color='#4f46e5')
    plt.plot(df_metrics['Dimensions'], df_metrics['Precision'], marker='s', lw=2, label='Precision', color='#10b981', alpha=0.7)
    plt.plot(df_metrics['Dimensions'], df_metrics['Recall'], marker='^', lw=2, label='Recall', color='#f59e0b', alpha=0.7)

    # Highlight the "Elbow Point" (Example: where gain starts to slow down)
    # Visual heuristic: Look at where the curve flattens
    plt.title("SVM Performance vs. Number of PCA Components", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Number of Principal Components kept as Input", fontsize=12)
    plt.ylabel("Metric Score (0.0 - 1.0)", fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('phase4_sensitivity_analysis.png', dpi=300)
    print("\n[SUCCESS] Final analysis plot saved as 'phase4_sensitivity_analysis.png'")
    
    # Conclusion
    best_n = df_metrics.loc[df_metrics['Accuracy'].idxmax(), 'Dimensions']
    print(f"\nPhase 4 Conclusion:")
    print(f"The best accuracy was achieved at {best_n} dimensions.")
    print("The 'Elbow Point' typically occurs around 30-50 dimensions, where most information is captured.")

if __name__ == "__main__":
    run_phase_4_optimization()
