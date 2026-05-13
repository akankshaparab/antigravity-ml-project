# SAGE: Cost-Efficient Semantic Query Routing and Optimization for Complex Text-to-SQL Systems

## Abstract
The efficacy of Text-to-SQL systems in Retrieval-Augmented Generation (RAG) architectures depends heavily on the geometric structure of high-dimensional query embedding spaces. This paper investigates the structural properties of these spaces, comparing the academic Spider benchmark (384D) with a production-scale environment (768D) from a Data Analyst agent from vanna.ai. Using Principal Component Analysis (PCA), we characterize the intrinsic dimensionality of the embedding manifold to identify regions of signal redundancy. This geometric assessment informs the development of a Query Routing Layer employing Support Vector Machines (SVM). By mapping cluster cohesion and identifying geometric dispersion among complex query types, this study demonstrates a principled approach to dimensionality reduction and automated query classification. The findings provide a technical framework for optimizing retrieval latency and infrastructure costs by strategically directing queries to Large Language Models based on their latent geometric characteristics, thereby improving the operational efficiency of enterprise Text-to-SQL pipelines.

**Keywords:** Retrieval-Augmented Generation (RAG), Text-to-SQL Systems, Dimensionality Reduction, Principal Component Analysis (PCA), Support Vector Machines (SVM), Query Routing Layer, Embedding Manifold Analysis

## 1 Introduction
- **Context:** The evolution of RAG-based Text-to-SQL systems has made natural language embeddings the standard for mapping user intent to database logic.
- **The Problem:** There is a lack of characterization regarding the geometric properties and signal redundancy within standard 384D/768D query embedding spaces.
- **Objective:** We propose using PCA to find intrinsic dimensionality and SVM to classify query complexity for optimized routing, reducing overhead without sacrificing accuracy.

## 2 Prior Research
### 2.1 Related Work
- **Spider Dataset**: The academic gold standard for Text-to-SQL benchmarks, providing a hierarchical difficulty classification (Easy to Extra Hard).
- **Embedding Models**: Modern focus on models like **BAAI/bge-small-en-v1.5**, which balance semantic density and compute efficiency.

### 2.2 Literature Review
- **PCA Foundations**: Exploration of PCA variants (Standard, Incremental, Sparse, Kernel) to handle large-scale embedding manifolds. Incremental PCA was used for RAM efficiency during production scaling.
- **SVM and Kernels**: Foundations of Support Vector Machines and the "Kernel Trick" for transforming non-linear semantic gaps into separable features.

## 3 Dataset
### 3.1 Data Description
- **Baseline Architecture**: The Spider dataset, comprising over 10,000 query-SQL pairs across 200 databases. Publicly available at https://yale-lily.github.io/spider.
- **Production Architecture**: A production-scale environment (768D) derived from a vanna.ai Data Analyst agent.

### 3.2 Data Statistics
The distribution of query difficulty (post-rebalancing) is as follows:
- **Easy**: 2,710
- **Medium**: 3,496
- **Hard**: 2,826
- **Extra Hard**: 661

### 3.3 EDA and Preprocessing
- **Difficulty Classifier Logic**: Implemented a point-based scoring system to quantify complexity:
    - **+1 Point**: `JOIN`, `GROUP BY`, `ORDER BY`, `HAVING`.
    - **+2 Points**: `INTERSECT`, `UNION`, `EXCEPT`.
    - **+2 Points**: Nested Queries (multiple `SELECT` statements).
- **Heuristic Rebalancing**: 
    - *Initial Model*: Scores of 1-2 were labeled as "Medium," leading to over-saturation of the class. 
    - *Optimized Model*: 0=Easy, 1=Medium, 2-3=Hard, and **>3=Extra Hard**. This shift corrected the bias where "Medium was catching too much" and "Extra Hard was catching too little."
- **Geometric Validation**: Vectors were normalized to unit length and verified using **Geometric Mean Squared Error (MSE)** to ensure directional consistency for the SVM decision boundaries.

## 4 Methodology
### 4.1 Dimensionality Reduction & Selection
We compared 4 PCA variants. Incremental PCA was prioritized for its RAM efficiency during production upserts.
- *[Image: pca_variant_comparison.png]*
- **Finding**: Approximately 80 dimensions were found to contribute 95% of the variance in the 384D baseline space.

### 4.2 Query Complexity Classification
- **Kernel Comparison**: Parallel comparison of RBF, Linear, and Poly kernels. The **RBF kernel** was selected as the winner for its superior ability to resolve linguistic overlap.
- **Weight Optimization**: Implemented `class_weight='balanced'` and custom adjusted weights to resolve the scarcity of Extra Hard queries.

## 5 Experiments
### 5.1 Experimental Configuration
- **Embedding Engine**: BAAI/bge-small-en-v1.5.
- **Data Split**: 80:20 train-test ratio using `stratify=y` to maintain the difficulty proportions across sets.

### 5.2 Hyperparameter Settings
- **Elbow Point Identification**: Determined the threshold where marginal returns on accuracy diminish.
- *[Image Placeholder: SVM Performance vs Number of PCA Components]*

### 5.3 Evaluation Metrics
Defined by **Precision, Recall, F1-Score**, and **Silhouette Score**. Special focus was placed on the transformation of Extra Hard recall from a floor of 32.58% to its production state.

## 6 Results and Discussion
### 6.1 Benchmark Results
#### 6.1.1 Intrinsic Dimensionality Analysis
Confirmed that the 768D manifold could be reduced to ~220 components (for production) or ~80 components (for baseline) while preserving necessary signal.
#### 6.1.2 Classification Performance
The optimized model resolved the semantic gap for 'Hard' queries (80% precision) and stabilized 'Extra Hard' classification.

### 6.2 Visualization Analysis
- **PCA vs. t-SNE**: PCA was used to capture global variance (difficulty mapping), while t-SNE was employed to capture local thematic neighborhoods.

### 6.3 Discussion
#### 6.3.1 Practical Implications
- **Query Routing Layer**: Enabled a dynamic production pipeline:
    - **"Easy" or "Medium"** queries are routed to faster, cost-effective models (**Gemini Flash**).
    - **"Hard" or "Extra Hard"** queries are directed to high-reasoning models (**Gemini Pro**).
- **Infrastructure ROI**: Achieved a **70.7% reduction in RAM-resident index size** on Pinecone, significantly lowering retrieval latency and monthly overhead.

#### 6.3.2 Identifying the Semantic Gap
Discusses how models encode linguistic meaning rather than database logic, requiring the weighted SVM to bridge the gap for "Extra Hard" queries.

## 7 Conclusion
A range of 50–80 PCA components is sufficient for representing Text-to-SQL complexity. The resulting SVM provides a principled foundation for automated enterprise query routing.

## 8 References
Citations for Spider Dataset, Scikit-learn, and BGE embedding models.
