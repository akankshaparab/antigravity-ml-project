# Outline

## 1 Introduction
- **Context:** Evolution of RAG-based Text-to-SQL systems and the role of natural language embeddings.
- **The Problem:** Lack of characterization regarding the geometric properties and signal redundancy within 384D/768D query embedding spaces.
- **Objective:** Using PCA to find intrinsic dimensionality and SVM to classify query complexity for optimized routing.

## 2 Prior Research
### Related Work
- Overview of the **Spider Dataset** as an academic gold standard for Text-to-SQL benchmarks.
- Current state of **Embedding Models** (specifically BAAI/bge-small-en-v1.5).

### Literature Review
- Theoretical foundations of **Principal Component Analysis (PCA)** and its variants (Standard, Incremental, Sparse, Kernel).
- **Support Vector Machines (SVM)** and the "Kernel Trick" for non-linear classification.

## 3 Dataset
### 3.1 Data Description
- **Baseline:** The Spider dataset (10,000+ pairs, 200 databases).
- **Production:** Production-scale environment (768D) from a vanna.ai Data Analyst agent.

### 3.2 Data Statistics
- **Distribution of query difficulty:** - Easy (2,710)
    - Medium (3,496)
    - Hard (2,826)
    - Extra Hard (661)

### 3.3 EDA and Preprocessing
- **Difficulty Classifier:** Logic based on SQL keyword count (JOIN, GROUP BY, etc.).
- **Geometric Validation:** Normalizing vectors to unit length for directional comparison.

## 4 Methodology
### 4.1 Dimensionality Reduction & Selection
- Comparison of 4 PCA variants based on RAM efficiency (Incremental) vs. interpretability (Sparse).
- *[Image Placeholder: pca_variant_comparison.png]*

### 4.2 Query Complexity Classification
- Implementing **SVC (Support Vector Classifier)** with RBF, Linear, and Poly kernels.
- Optimization of hyperparameters: Setting `class_weight='balanced'` to handle Extra Hard query scarcity.

## 5 Experiments
### 5.1 Experimental Configuration
- **Model:** BAAI/bge-small-en-v1.5.
- **Split:** 80:20 train-test ratio with `stratify=y` to maintain difficulty proportions.

### 5.2 Hyperparameter Settings
- Identification of the **"Elbow Point"**: Determining the threshold where marginal returns on accuracy diminish.
- *[Image Placeholder: SVM Performance vs Number of PCA Components]*

### 5.3 Evaluation Metrics
- Mathematical definitions of **Precision, Recall, F1-Score**, and **Silhouette Score**.

## 6 Results and Discussion
### 6.1 Benchmark Results
#### 6.1.1 Intrinsic Dimensionality Analysis
- Finding that ~80 dimensions contribute 95% of the variance in a 384D space.

#### 6.1.2 Classification Performance
- High precision for 'Hard' queries (80%) but low recall for 'Extra Hard' (32.58%) due to semantic gaps.

### 6.2 Visualization Analysis
- **PCA (Linear) vs. t-SNE (Non-linear):** PCA captures global variance (difficulty) while t-SNE captures local neighbourhoods (themes).

### 6.3 Discussion
#### 6.3.1 Practical Implications
- **Query Routing Layer:** Directing simple queries to smaller models and complex queries to advanced LLMs.
- **Infrastructure ROI:** Reducing Pinecone storage and retrieval latency.

#### 6.3.2 Identifying the Semantic Gap
- The model struggles with "Extra Hard" because it encodes linguistic meaning rather than database logic.

## 7 Conclusion
- **Summary:** 50–80 PCA components are sufficient for representing Text-to-SQL complexity.
- The SVM provides a principled foundation for automated enterprise query routing.

## 8 References
- Standard citations for Spider Dataset, Scikit-learn, and BGE embedding models.
