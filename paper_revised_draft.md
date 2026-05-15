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
- **Baseline Architecture**: The Spider dataset, comprising approximately 10,000 query-SQL pairs across 200 databases. Publicly available at https://yale-lily.github.io/spider.
- **Production Architecture**: A production-scale environment (768D) derived from a vanna.ai Data Analyst agent.

### 3.2 Data Statistics
The distribution of query difficulty (post-rebalancing) is as follows:
- **Easy**: 2,710
- **Medium**: 3,496
- **Hard**: 2,826
- **Extra Hard**: 661

![Spider Class Distribution](spider_class_distribution.png)
*Figure 1: Spider Class Distribution — Visualizing the inherent data imbalance toward Medium and Hard complexity levels.*

#### 3.2.2 Production Environment (Pinecone Database)
The production environment integrates the baseline Spider dataset with real-world query data fetched from the vanna.ai Data Analyst agent. The combined manifold contains:
- **Spider Baseline**: 9,693 queries
- **Live Production Queries**: 1,742 queries
- **Total Vector Capacity**: 11,435 observations (768D)

This expanded dataset ensures that the SVM-RBF router is trained not only on academic benchmarks but also on the specific semantic nuances of live enterprise traffic.

![Pinecone Data Distribution](produc_vers/pinecone_distribution.png)
*Figure 2: Pinecone Data Distribution — Integration of 1,742 live enterprise queries with the Spider baseline manifold.*

### 3.3 EDA and Preprocessing
- **Data Matrix Formatting**: The dataset was prepared in standard machine learning format: a feature matrix $X$ (embeddings) and a label vector $y$ (difficulty classes). Matrix $X$ is used for training and testing, while $y$ serves as the ground truth for prediction.
- **Geometric Retrieval Theory**: By setting `normalize_embeddings=True` during the embedding process, we constrained vectors to a constant magnitude (unit length). Since vectors consist of both magnitude and direction, this normalization allows the model to prioritize **directional similarity**. Vectors pointing in the same direction indicate a similar difficulty level, providing a more robust metric than distance alone in high-dimensional space.

![Geometric Cluster Map](phase3_geometric_clusters.png)
*Figure 3: Geometric Cluster Map — 2D projection confirming that query difficulty levels form distinct, separable semantic neighborhoods.*
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

![PCA Variant Comparison](pca_variant_comparison.png)
*Figure 4: PCA Variant Comparison — Benchmark of training time and variance retention across four major dimensionality reduction methods.*
- **Component Rationale**: We standardized the initial analysis at **50 components** for three primary reasons:
    - **Benchmarking Efficiency**: It provided a manageable baseline for comparing the computational overhead of the four PCA variants.
    - **Operational Speed**: Lower dimensionality is critical for the real-time throughput of the SVM in a production query routing layer.
    - **Information Density**: Initial scree plot analysis suggested a primary elbow point at 50, where the "vast majority" of the semantic meaning was captured, despite a lower total variance explained (~62.5%).

![Variance Analysis Plot](variance_analysis_plot.png)
*Figure 5: Variance Analysis Plot — Quantitative proof of the 62.5% vs. 90% variance thresholds across component counts.*
- **Scree Plot Insights**: The scree plot was utilized to answer how variance is distributed across dimensions within the reduced space, helping to define the threshold where marginal gains in signal diminish.

![Baseline Scree Plot](phase3_scree_plot_final.png)
*Figure 6: Baseline Scree Plot — Identifying the 50-component primary elbow point in the 384D academic manifold.*

![Production Scree Plot](produc_vers/scree_plot_768.png)
*Figure 7: Production Scree Plot — Variance distribution across the 768D production-scale embedding manifold.*

### 4.2 Query Complexity Classification
- **Kernel Comparison**: Parallel comparison of RBF, Linear, and Poly kernels. The **RBF kernel** was selected as the winner for its superior ability to resolve **semantic ambiguity** within the query space.

![Kernel Comparison Graph](phase4_kernel_comparison.png)
*Figure 8: Kernel Comparison Graph — Visualizing the RBF kernel's superior ability to resolve non-linear semantic boundaries.*
- **Weight Optimization**: Implemented `class_weight='balanced'` and custom adjusted weights to resolve the scarcity of Extra Hard queries.

## 5 Experiments
### 5.1 Experimental Configuration
- **Embedding Engine**: BAAI/bge-small-en-v1.5.
- **Data Split**: 80:20 train-test ratio using `stratify=y` to maintain the difficulty proportions across sets.

### 5.2 Hyperparameter Settings
- **Identification of the Elbow Zone**: Through sensitivity analysis, we identified that the optimal performance-to-cost ratio occurs in an **"Elbow Zone" between 30 and 50 dimensions**. Beyond this point, gains in accuracy, precision, and recall were found to plateau.

![SVM Sensitivity Analysis (Production)](produc_vers/sensitivity_results.png)
*Figure 9: SVM Sensitivity Results — Performance plateau confirming the 30-50D range as the optimal efficiency zone.*

### 5.3 Evaluation Metrics
Performance was evaluated using several statistical indicators:
- **Weighted Metrics**: We calculated weighted averages for **Accuracy, Precision, and Recall**. This weighting is essential as it accounts for the relative size (support) of each difficulty set, ensuring that the dominant "Easy/Medium" classes do not overshadow the "Extra Hard" minority.
- **Silhouette Score**: This metric was used to measure cluster cohesion. A positive silhouette score indicated that queries of the same type form coherent geometric clusters, validating the semantic separation of the difficulty labels.
- **Heatmap Insights**: The generated similarity heatmap showed distinct diagonal blocks, indicating high intra-class similarity. Notably, the "Extra Hard" block appeared the most isolated, confirming it as a distinct semantic neighborhood. The heatmap also revealed that while difficulty drives separation, secondary clustering often occurs based on **thematic similarity**.

![Similarity Heatmap](phase3_similarity_heatmap.png)
*Figure 10: Similarity Heatmap — Demonstrating high intra-class similarity and clear diagonal cluster separation.*

## 6 Results and Discussion
### 6.1 Benchmark Results and Evolution

#### 6.1.1 Initial Baseline Performance (Standardized at 50 Components)
The first iteration of the classification layer utilized the 50-component subspace. While efficient, this configuration revealed critical limitations in handling high-complexity queries.

- **Statistical Imbalance**: The model initially applied equal weighting to all classes. Given the skewed distribution of the dataset (dominated by "Medium" and "Hard"), the minority "Extra Hard" class was poorly learned.
- **Information Loss in the "Tail"**: At 50 components, only ~62.5% of the total variance was preserved. Analysis revealed that complex SQL keywords are often encoded in the "tail" of the embedding variance; by truncating at 50, these logical cues were discarded in favor of general semantic themes.
- **Evaluation Visuals (Initial Phase)**:

![Baseline Confusion Matrix](pre_vers_confu_matr.png)
*Figure 11: Baseline Confusion Matrix — Illustrating the misclassification of Extra Hard queries in the 50D baseline model.*

    - **Observation**: The model struggled significantly with "Extra Hard" queries (only 43/132 correct). There was a noticeable "pull" toward the Medium class, indicating a systemic bias toward predicting the majority categories.

![Baseline Metric Comparison](pre_vers_metric_comp.png)
*Figure 12: Baseline Metric Comparison — Precision and recall disparities across difficulty tiers in the aggressive compression phase.*

    - **Observation**: Performance was highly uneven. While the model achieved a respectable 80% precision for "Hard" queries, it hit a performance floor of **32% recall** for "Extra Hard" types.
- **Conclusion**: Raw embeddings and aggressive compression were insufficient for production. The near-random performance on complex queries confirmed that class imbalance and information loss must be addressed simultaneously.

#### 6.1.2 Optimization Phase: Increasing Manifold Fidelity
To bridge the gap between English phrasing and SQL complexity, two primary changes were implemented:
1. **Expansion of Dimensionality**: The subspace was expanded from 50 to **179 components**, increasing variance retention from 62.5% to **90%**. This preserved the subtle linguistic cues necessary for logical mapping.
2. **Balanced Class Weighting**: Custom weights were introduced to the SVM to ensure the minority "Extra Hard" class was absorbed with higher sensitivity.

#### 6.1.3 Final Production Evaluation (Optimized State)
The optimized model was re-evaluated against the 179D baseline manifold and subsequently scaled to the **220D production manifold** to maintain 90% variance retention for the 768D production embeddings.

- **Evaluation Visuals (Production Phase)**:

![Production Confusion Matrix](phase5_confusion_matrix.png)
*Figure 13: Production Confusion Matrix — Successful structural learning and diagonal performance in the optimized manifold (179D Baseline / 220D Production).*

    - **Observation**: The classifier achieved strong diagonal performance across all four classes. The most notable remaining confusion was a minor overlap between "Medium" and "Easy," suggesting a high semantic similarity between adjacent complexity levels.
    - **Success with Complexity**: Accuracy on "Extra Hard" queries jumped to **98/132**, validating that the model successfully learned structural differences.

![Production Metric Comparison](phase5_metric_comparison.png)
*Figure 14: Production Metric Comparison — Uniform performance across all query classes in the final 220D production layer.*

    - **Observation**: The "short bars" of the baseline were replaced by strong, uniform precision and recall across the board.
- **Conclusion**: By expanding the manifold to **179 components** (baseline) or **220 components** (production) to capture 90% variance, and balancing the classifier's sensitivity, the system became viable for production-grade routing.

![Live vs. Spider Distribution Comparison](produc_vers/live_vs_spider_comparison.png)
*Figure 15: Live vs. Baseline Projection — Manifold comparison confirming academic benchmark compatibility with production traffic.*

### 6.2 Visualization Analysis
- **PCA vs. t-SNE**: PCA was used to capture global variance (difficulty mapping), while t-SNE was employed to capture local thematic neighborhoods.

### 6.3 Discussion
#### 6.3.1 Implications for Production Architecture
The empirical results provide a robust, evidence-backed justification for the deployment of **Standard PCA** within the SAGE production pipeline. The following architectural implications were identified:

- **Efficient Query Routing Layer**: The implementation of the SVM-RBF classifier enables a high-efficiency model cascading strategy. 
    - **Tier 1 (High Volume)**: "Easy" and "Medium" queries, which constitute approximately **80% of total traffic**, are successfully routed to **free models via OpenRouter**.
    - **Tier 2 (High Logic)**: "Hard" and "Extra Hard" queries are strategically directed to **Claude Haiku**, ensuring that premium compute is only utilized for tasks requiring advanced logical reasoning.
- **Infrastructure ROI and Performance**: The discovery of significant signal redundancy within the production manifold allows for the elimination of semantic noise. This reduction directly translates to **accelerated similarity search retrieval** and a minimized memory footprint, ensuring the system remains responsive at production scales while minimizing cloud overhead (see Table 1).

**Table 1: Infrastructure ROI via Dimensionality Reduction**
| Feature | Original (Without PCA) | Reduced (With PCA) | Savings |
| :--- | :--- | :--- | :--- |
| **Dimensions** | 768 | 220 | ~71% Reduction |
| **Data Size (Est.)** | 37.2 MB | 10.9 MB | 26.3 MB Saved |


#### 6.3.2 Identifying the Semantic Gap
Analysis of the misclassifications reveals that the core challenge lies in the model's sensitivity: it is naturally more attuned to **semantic themes** (the subject of the query) than to **keyword complexity** (the structure of the query). When aggressive PCA compression is applied, the structural logic is the first to be discarded as "noise." By expanding the **baseline architecture to 179 components** and applying balanced weights, we successfully bridged the gap between the natural language meaning and the underlying SQL logical structure.

![Semantic Blind Spot Map](produc_vers/blind_spot_map.png)
*Figure 16: Semantic Blind Spot Map — Visualizing geometric regions where structural logic is susceptible to high-dimensional semantic noise.*

## 7 Conclusion
The research demonstrates that while an initial elbow point of **50 PCA components** captures the semantic core of Text-to-SQL queries, an expansion to **179 components** (baseline) or **220 components** (production) is necessary to preserve the logical "tail" required for complex query classification. The resulting SVM-RBF model provides a principled, cost-efficient foundation for automated enterprise query routing.

## 8 References
1. **Yu, T., et al. (2018).** "Spider: A Large-Scale Hierarchical Semantic Parsing and Text-to-SQL Dataset." *arXiv preprint arXiv:1809.08887*.
2. **Xiao, S., et al. (2023).** "C-Pack: Packaged Resources to Advance General Chinese Embedding." *arXiv preprint arXiv:2309.07597*. (BGE Embedding Models)
3. **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*.
4. **Salton, G., et al. (1975).** "A Vector Space Model for Automatic Indexing." *Communications of the ACM*.
5. **Conneau, A., et al. (2018).** "What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties." *ACL*.
6. **Levina, E., & Bickel, P. (2004).** "Maximum Likelihood Estimation of Intrinsic Dimension." *NIPS*.
7. **Gemini Team, Google. (2023).** "Gemini: A Family of Highly Capable Multimodal Models."
8. **Pinecone Systems Inc. (2024).** "Pinecone Vector Database Service."
9. **Van der Maaten, L., & Hinton, G. (2008).** "Visualizing Data using t-SNE." *Journal of Machine Learning Research*.
10. **Vanna.ai (2024).** Documentation for Production Data Analyst Agent Framework.


---
**Disclosure:** This document was drafted and structured with the assistance of AI tools for linguistic refinement and technical organization. Final data validation and architectural decisions were performed by the author.
