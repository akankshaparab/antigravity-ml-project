import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. Load Data
print("Loading 768D embeddings...")
data = np.load('spider_768_embeddings.npz')
X_full = data['X']
y = data['y']

X_train_f, X_test_f, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# 2. Sensitivity Test: PCA Components
pca_ranges = [50, 100, 200, 300, 400, 500, 768]
f1_results = []

print("\n--- Running Sensitivity Analysis (PCA Components) ---")
for n in pca_ranges:
    # Reduce dimensions
    pca = PCA(n_components=n).fit(X_train_f)
    X_train = pca.transform(X_train_f)
    X_test = pca.transform(X_test_f)
    
    # Train SVM
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    score = f1_score(y_test, y_pred, average='weighted')
    f1_results.append(score)
    print(f"PCA Components: {n} | Weighted F1: {score:.4f}")

# 3. Sensitivity Test: SVM 'C' (Penalty Parameter)
c_ranges = [0.1, 1.0, 10, 100]
c_results = []

print("\n--- Running Sensitivity Analysis (SVM C-Parameter) ---")
# Using the optimal PCA from step 2 (e.g., 300)
pca = PCA(n_components=300).fit(X_train_f)
X_train_opt = pca.transform(X_train_f)
X_test_opt = pca.transform(X_test_f)

for c_val in c_ranges:
    clf = SVC(kernel='rbf', C=c_val)
    clf.fit(X_train_opt, y_train)
    y_pred = clf.predict(X_test_opt)
    
    score = f1_score(y_test, y_pred, average='weighted')
    c_results.append(score)
    print(f"SVM C={c_val} | Weighted F1: {score:.4f}")

# 4. Plotting Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(pca_ranges, f1_results, marker='o', color='b')
plt.title('Accuracy vs PCA Components')
plt.xlabel('Dimensions')
plt.ylabel('F1 Score')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([str(c) for c in c_ranges], c_results, marker='s', color='r')
plt.title('Accuracy vs SVM C-Value')
plt.xlabel('C Value')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.savefig('sensitivity_results.png')
print("\nSensitivity results saved to sensitivity_results.png")
