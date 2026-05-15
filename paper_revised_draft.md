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
- **Data Matrix Formatting**: The dataset was prepared in standard machine learning format: a feature matrix $X$ (embeddings) and a label vector $y$ (difficulty classes). Matrix $X$ is used for training and testing, while $y$ serves as the ground truth for prediction.
- **Geometric Retrieval Theory**: By setting `normalize_embeddings=True` during the embedding process, we constrained vectors to a constant magnitude (unit length). Since vectors consist of both magnitude and direction, this normalization allows the model to prioritize **directional similarity**. Vectors pointing in the same direction indicate a similar difficulty level, providing a more robust metric than distance alone in high-dimensional space.
- **Difficulty Classifier Logic**: Implemented a point-based scoring system to quantify complexity:
    - **+1 Point**: `JOIN`, `GROUP BY`, `ORDER BY`, `HAVING`.
    - **+2 Points**: `INTERSECT`, `UNION`, `EXCEPT`.
    - **+2 Points**: Nested Queries (multiple `SELECT` statements).
- **Heuristic Rebalancing**: 
    - *Initial Model*: Scores of 1-2 were labeled as "Medium," leading to over-saturation of the class. 
    - *Optimized Model*: 0=Easy, 1=Medium, 2-3=Hard, and **>3=Extra Hard**. This shift corrected the bias where "Medium was catching too much" and "Extra Hard was catching too little."
- **Geometric Validation**: Vectors were normalized to unit length and verified using **Geometric Mean Squared Error (MSE)**. For visualization purposes, string labels were converted to a numerical format (`Easy: 0` to `Extra Hard: 3`) to facilitate consistent color coding across projections.

## 4 Methodology
### 4.1 Dimensionality Reduction & Selection
- **Benchmarking Methodology**: We evaluated four PCA variants (Standard, Incremental, Sparse, Kernel) to identify the most efficient method for production scaling. The `benchmark_pca()` function used a timing mechanism (`time.time()`) to record the duration before and after the `fit_transform()` operation, identifying Standard PCA as the optimal balance between speed and variance retention.
- **Component Rationale**: For the purpose of comparing these variants, we standardized the analysis at **50 components**. This number was chosen as a representative baseline where the "elbow" of the variance curve typically begins, allowing for a fair comparison of computational overhead across methods.
- **Scree Plot Insights**: The scree plot was utilized to answer how variance is distributed across dimensions within the reduced space, helping to define the threshold where marginal gains in signal diminish.
- *[Image: pca_variant_comparison.png]*

### 4.2 Query Complexity Classification
- **Kernel Comparison**: Parallel comparison of RBF, Linear, and Poly kernels. The **RBF kernel** was selected as the winner for its superior ability to resolve linguistic overlap.
- **Weight Optimization**: Implemented `class_weight='balanced'` and custom adjusted weights to resolve the scarcity of Extra Hard queries.

## 5 Experiments
### 5.1 Experimental Configuration
- **Embedding Engine**: BAAI/bge-small-en-v1.5.
- **Data Split**: 80:20 train-test ratio using `stratify=y` to maintain the difficulty proportions across sets.

### 5.2 Hyperparameter Settings
- **Identification of the Elbow Zone**: Through sensitivity analysis, we identified that the optimal performance-to-cost ratio occurs in an **"Elbow Zone" between 30 and 50 dimensions**. Beyond this point, gains in accuracy, precision, and recall were found to plateau.
- *[Image Placeholder: SVM Performance vs Number of PCA Components]*

### 5.3 Evaluation Metrics
Performance was evaluated using several statistical indicators:
- **Weighted Metrics**: We calculated weighted averages for **Accuracy, Precision, and Recall**. This weighting is essential as it accounts for the relative size (support) of each difficulty set, ensuring that the dominant "Easy/Medium" classes do not overshadow the "Extra Hard" minority.
- **Silhouette Score**: This metric was used to measure cluster cohesion. A positive silhouette score indicated that queries of the same type form coherent geometric clusters, validating the semantic separation of the difficulty labels.
- **Heatmap Insights**: The generated similarity heatmap showed distinct diagonal blocks, indicating high intra-class similarity. Notably, the "Extra Hard" block appeared the most isolated, confirming it as a distinct semantic neighborhood. The heatmap also revealed that while difficulty drives separation, secondary clustering often occurs based on thematic/theme similarity.

## 6 Results and Discussion
### 6.1 Benchmark Results
#### 6.1.1 Evaluation Workflow
The final evaluation followed a rigorous pipeline:
1.  **Data Partitioning**: An 80:20 stratified train-test split was used to maintain the proportional representation of all four difficulty levels.
2.  **Compression**: The dimensions were reduced to the optimal **50-component** threshold identified in the elbow analysis.
3.  **Training**: An RBF-SVM was trained on this 50D space using `class_weight='balanced'`.
4.  **Prediction and Confidence**: The model made predictions on the unseen test set, which were compared against true labels to measure performance and calculate confidence scores.

#### 6.1.2 Classification Performance
The optimized model resolved the semantic gap for 'Hard' queries (80% precision) and stabilized 'Extra Hard' classification. The tabulated results provide a structured numerical summary of the model's strengths across different complexity tiers.



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
