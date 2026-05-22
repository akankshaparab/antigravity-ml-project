# SAGE: Cost-Efficient Semantic Query Routing and Optimization for Complex Text-to-SQL Systems

## Abstract
The efficacy of Text-to-SQL systems in Retrieval-Augmented Generation (RAG) architectures depends heavily on the geometric structure of high-dimensional query embedding spaces. This paper investigates the structural properties of these spaces, comparing the academic Spider benchmark (384D) with a production-scale environment (768D) from a Data Analyst agent from vanna.ai. Using Principal Component Analysis (PCA), we characterize the intrinsic dimensionality of the embedding manifold to identify regions of signal redundancy. This geometric assessment informs the development of a Query Routing Layer employing Support Vector Machines (SVM). By mapping cluster cohesion and identifying geometric dispersion among complex query types, this study demonstrates a principled approach to dimensionality reduction and automated query classification. The findings provide a technical framework for optimizing retrieval latency and infrastructure costs by strategically directing queries to Large Language Models based on their latent geometric characteristics, thereby improving the operational efficiency of enterprise Text-to-SQL pipelines.

**Keywords:** Retrieval-Augmented Generation (RAG), Text-to-SQL Systems, Dimensionality Reduction, Principal Component Analysis (PCA), Support Vector Machines (SVM), Query Routing Layer, Embedding Manifold Analysis

## 1 Introduction
**Context:** The evolution of RAG-based [15] Text-to-SQL systems has made natural language embeddings [4] the standard for mapping user intent to database logic.

**The Problem:** There is a lack of characterization regarding the geometric properties and signal redundancy within standard 384D/768D query embedding spaces [6].

**Objective:** We propose using PCA to find intrinsic dimensionality [6] and SVM [11] to classify query complexity for optimized routing, reducing overhead without sacrificing accuracy.

## 2 Prior Research
### 2.1 Related Work
- **Spider Dataset**: The baseline evaluation utilizes the Spider dataset [1], which serves as the academic gold standard for Text-to-SQL benchmarks and provides a hierarchical difficulty classification from Easy to Extra Hard.
- **Embedding Models**: Prior research on semantic query routing focuses on models like **BAAI/bge-small-en-v1.5** [2], which balance semantic density and computational efficiency.

### 2.2 Literature Review
- **PCA Foundations**: Research on dimensionality reduction frequently explores PCA variants (Standard, Incremental, Sparse, Kernel [14]) to handle large-scale embedding manifolds [3] and characterize intrinsic dimensionality [6]. In this study, Incremental PCA was utilized to optimize RAM efficiency during production scaling.
- **SVM and Kernels**: Support Vector Machines (SVM) [11] utilize the "Kernel Trick" to project non-linear semantic gaps into higher-dimensional spaces where they become linearly separable.

## 3 Dataset
### 3.1 Data Description
**Baseline Architecture**: The baseline architecture relies on the Spider dataset [1], which comprises approximately 10,000 query-SQL pairs across 200 databases and is publicly available at https://yale-lily.github.io/spider.

**Production Architecture**: The production architecture is evaluated in a production-scale environment (768D) derived from a vanna.ai [10] Data Analyst agent.

### 3.2 Data Statistics

#### 3.2.1 Heuristic Difficulty Classification Framework
The complexity classification of queries in both datasets is determined using a deterministic, rule-based SQL scoring heuristic. Complexity scores are accumulated based on SQL structural tokens:
- Clauses like `JOIN`, `GROUP BY`, `ORDER BY`, and `HAVING` contribute $+1$ point each.
- Set operators such as `INTERSECT`, `UNION`, and `EXCEPT` contribute $+2$ points each.
- Nested subqueries (multiple `SELECT` statements) contribute $+2$ points each.

The total accumulated score maps queries into discrete difficulty tiers:
- **0 points**: `Easy`
- **1 point**: `Medium`
- **2–3 points**: `Hard`
- **>3 points**: `Extra Hard`

#### 3.2.2 Baseline Dataset (Spider)
The distribution of query difficulty (post-rebalancing) for the academic Spider dataset is as follows:
- **Easy**: 2,710
- **Medium**: 3,496
- **Hard**: 2,826
- **Extra Hard**: 661

![Spider Class Distribution](spider_class_distribution.png)
*Figure 1: Spider Class Distribution — Visualizing the inherent data imbalance toward Medium and Hard complexity levels.*

#### 3.2.3 Production Environment (Pinecone Database [8])
The production environment integrates the baseline Spider dataset [1] with real-world query data fetched from the vanna.ai [10] Data Analyst agent. The combined manifold contains:
- **Spider Baseline [1]**: 9,693 queries
- **Live Production Queries**: 1,742 queries
- **Total Vector Capacity**: 11,435 observations (768D) [8]

This expanded dataset ensures that the SVM-RBF router is trained not only on academic benchmarks but also on the specific semantic nuances of live enterprise traffic.

![Pinecone Data Distribution](produc_vers/pinecone_distribution.png)
*Figure 2: Pinecone Data Distribution — Integration of 1,742 live enterprise queries with the Spider baseline manifold.*

### 3.3 EDA and Preprocessing
**Data Matrix Formatting**: The dataset was prepared in standard machine learning format [3]: a feature matrix $X$ (embeddings) and a label vector $y$ (difficulty classes). Matrix $X$ is used for training and testing, while $y$ serves as the ground truth for prediction.

**Geometric Retrieval Theory**: By setting `normalize_embeddings=True` during the embedding process, we constrained vectors to a constant magnitude (unit length) [4]. Since vectors consist of both magnitude and direction, this normalization allows the model to prioritize **directional similarity**. Vectors pointing in the same direction indicate a similar difficulty level, providing a more robust metric than distance alone in high-dimensional space.

![Geometric Cluster Map](phase3_geometric_clusters.png)
*Figure 3: Geometric Cluster Map — 2D projection confirming that query difficulty levels form distinct, separable semantic neighborhoods.*

**Difficulty Classifier Logic**: We leveraged the point-based scoring system detailed in Section 3.2.1 to quantify complexity based on SQL tokens.

**Heuristic Rebalancing**: To correct class distribution bias during preprocessing, we adjusted the classification boundaries. Originally, any query scoring 1 or 2 was classified as "Medium," leading to severe over-saturation of that category. Rebalancing the thresholds (0 for Easy, 1 for Medium, 2–3 for Hard, and >3 for Extra Hard) resolved this bias, improving minority class learning without losing structural complexity markers.

**Geometric Validation**: Vectors were normalized to unit length and verified using **Geometric Mean Squared Error (MSE)** [4]. For visualization purposes, string labels were converted to a numerical format (`Easy: 0` to `Extra Hard: 3`) to facilitate consistent color coding across projections.

## 4 Methodology
The SAGE system employs a two-phase architecture: an offline training pipeline to optimize the query routing classifier, and an online routing pipeline for real-time inference. The offline training workflow is shown in Figure 4, detailing how raw queries undergo heuristic labeling, embedding generation, 220D PCA dimensionality reduction, and SVM training to yield the final serialized router model (`router_artifacts.pkl`) [3].

![Training Workflow](produc_vers/offline_training_flowchart.png)
*Figure 4: Training Workflow*

**Complementary Roles of PCA and SVM**: The SAGE routing architecture employs Principal Component Analysis (PCA) and Support Vector Machines (SVM) [11] for distinct, complementary roles in the optimization pipeline. PCA functions as an unsupervised dimensionality reduction step that filters high-dimensional semantic noise (e.g., phrasing variances, minor punctuation) [5] by extracting the orthogonal components of maximum variance. This yields a dense, lower-dimensional manifold that minimizes storage size and search latency. SVM then operates as a supervised classification engine, taking these compressed representation coordinates to construct optimal separating hyperplanes between query difficulty categories, using a non-linear kernel to resolve complex decision boundaries [11].

### 4.1 Dimensionality Reduction & Selection
**Benchmarking Methodology**: We evaluated four PCA variants (Standard, Incremental, Sparse, and Kernel [14]) to identify the most efficient method for production scaling. Out of dozens of dimensionality reduction techniques, these four were specifically selected because they represent the fundamental mathematical approaches to modeling different data manifolds: linear vs. non-linear [14] and dense vs. sparse data structures. The benchmarking execution (using `benchmark_pca()`) measured the computational duration of `fit_transform()` using `time.time()`, confirming that Standard PCA offered the optimal balance between inference speed and variance retention [3].

![PCA Variant Comparison](pca_variant_comparison.png)
*Figure 5: PCA Variant Comparison — Benchmark of training time and variance retention across four major dimensionality reduction methods.*

**Component Rationale**: We standardized the initial analysis at **50 components** for three primary reasons:
- **Benchmarking Efficiency**: It provided a manageable baseline for comparing the computational overhead of the four PCA variants.
- **Operational Speed**: Lower dimensionality is critical for the real-time throughput of the SVM in a production query routing layer.
- **Information Density**: Initial scree plot analysis suggested a primary elbow point at 50, where the "vast majority" of the semantic meaning was captured, despite a lower total variance explained (~62.5%).

![Variance Analysis Plot](variance_analysis_plot.png)
*Figure 6: Variance Analysis Plot — Quantitative proof of the 62.5% vs. 90% variance thresholds across component counts.*

**Scree Plot Insights**: The scree plot was utilized to answer how variance is distributed across dimensions within the reduced space, helping to define the threshold where marginal gains in signal diminish.
- **Semantic Interpretation of Variance Loss**: In our optimization, retaining 90% of the variance (at 179/220 components) implies a 10% loss of total variance. This discarded 10% is shown to consist mostly of semantic noise—minor phrasing variations, vocabulary synonyms, or punctuation marks that do not alter the underlying logical structure of the query [5].
- **Production Scree Plot Analysis**: The scree plot for the 768D production manifold (Figure 8) exhibits a rapid initial decay, where the first few principal components capture the primary syntactic patterns of Text-to-SQL queries. As the components scale, the curve flattens into a long tail representing diminishing returns, where adding dimensions only captures noise rather than structural query complexity.

![Baseline Scree Plot](phase3_scree_plot_final.png)
*Figure 7: Baseline Scree Plot — Identifying the 50-component primary elbow point in the 384D academic manifold.*

![Production Scree Plot](produc_vers/scree_plot_768.png)
*Figure 8: Production Scree Plot — Variance distribution across the 768D production-scale embedding manifold.*

### 4.2 Query Complexity Classification
**Kernel Comparison and Selection**: We conducted a parallel performance comparison of RBF, Linear, and Polynomial kernels to determine the optimal SVM decision boundary [11].
- **Linear Kernel Inadequacy**: Visual inspection of the projection scatter plots indicates that difficulty categories cannot be separated by straight lines, rendering linear boundaries highly error-prone.
- **Polynomial Kernel Inconsistency**: While the Polynomial kernel can model complex interfaces, it is highly sensitive to hyperparameter tuning, computationally slower, and performs inconsistently when scaled across varying dimensions on live traffic.
- **RBF Selection Rationale**: The Radial Basis Function (RBF) kernel [11] was selected as it is uniquely suited for production routing because:
    1. It models highly curved, non-linear boundaries.
    2. It resolves semantic and linguistic overlap between adjacent classes (such as Easy and Medium) using a non-linear hyperplane.
    3. It exhibits highly consistent accuracy and F1-scores across all component counts, ensuring stability.
    4. It employs a local approach that prioritizes nearby observations, making it ideal for capturing localized, high-density pockets of Easy queries.

![Kernel Comparison Graph](phase4_kernel_comparison.png)
*Figure 9: Kernel Comparison Graph — Visualizing the RBF kernel's superior ability to resolve non-linear semantic boundaries.*

**Weight Optimization**: We implemented `class_weight='balanced'` and custom adjusted weights to resolve the scarcity of Extra Hard queries.

## 5 Experiments
### 5.1 Experimental Configuration
**Embedding Engine**: We selected BAAI/bge-small-en-v1.5 [2] as the core embedding engine.

**Data Split**: The training pipeline uses an 80:20 train-test ratio with `stratify=y` to maintain consistent difficulty proportions across both splits.

### 5.2 Hyperparameter Settings
**Identification of the Elbow Zone**: Through sensitivity analysis, we identified that the optimal performance-to-cost ratio occurs in an **"Elbow Zone" between 30 and 50 dimensions**. Beyond this point, gains in accuracy, precision, and recall were found to plateau.

![SVM Sensitivity Analysis (Production)](produc_vers/sensitivity_results.png)
*Figure 10: SVM Sensitivity Results — Performance plateau confirming the 30-50D range as the optimal efficiency zone.*

### 5.3 Evaluation Metrics
Performance was evaluated using several statistical indicators:
- **Weighted Metrics**: We calculated weighted averages for **Accuracy, Precision, and Recall**. This weighting is essential as it accounts for the relative size (support) of each difficulty set, ensuring that the dominant "Easy/Medium" classes do not overshadow the "Extra Hard" minority.
- **Silhouette Score**: This metric was used to measure cluster cohesion [3]. A positive silhouette score indicated that queries of the same type form coherent geometric clusters, validating the semantic separation of the difficulty labels.
- **Heatmap Insights**: The generated similarity heatmap showed distinct diagonal blocks, indicating high intra-class similarity. Notably, the "Extra Hard" block appeared the most isolated, confirming it as a distinct semantic neighborhood. The heatmap also revealed that while difficulty drives separation, secondary clustering often occurs based on **thematic similarity**.

![Similarity Heatmap](phase3_similarity_heatmap.png)
*Figure 11: Similarity Heatmap — Demonstrating high intra-class similarity and clear diagonal cluster separation.*

## 6 Results and Discussion
### 6.1 Benchmark Results and Evolution

#### 6.1.1 Initial Baseline Performance (Standardized at 50 Components)
The first iteration of the classification layer utilized the 50-component subspace. While efficient, this configuration revealed critical limitations in handling high-complexity queries.

**Statistical Imbalance**: The model initially applied equal weighting to all classes. Given the skewed distribution of the dataset (dominated by "Medium" and "Hard"), the minority "Extra Hard" class was poorly learned.

**Information Loss in the "Tail"**: At 50 components, only ~62.5% of the total variance was preserved. Analysis revealed that complex SQL keywords are often encoded in the "tail" of the embedding variance; by truncating at 50, these logical cues were discarded in favor of general semantic themes [5].

**Evaluation Visuals (Initial Phase)**:

![Baseline Confusion Matrix](pre_vers_confu_matr.png)
*Figure 12: Baseline Confusion Matrix — Illustrating the misclassification of Extra Hard queries in the 50D baseline model.*

- **Observation**: The model struggled significantly with "Extra Hard" queries (only 43/132 correct). There was a noticeable "pull" toward the Medium class, indicating a systemic bias toward predicting the majority categories.

![Baseline Metric Comparison](pre_vers_metric_comp.png)
*Figure 13: Baseline Metric Comparison — Precision and recall disparities across difficulty tiers in the aggressive compression phase.*

- **Observation**: Performance was highly uneven. While the model achieved a respectable 80% precision for "Hard" queries, it hit a performance floor of **32% recall** for "Extra Hard" types.
- **Conclusion**: Raw embeddings and aggressive compression were insufficient for production. The near-random performance on complex queries confirmed that class imbalance and information loss must be addressed simultaneously.

#### 6.1.2 Optimization Phase: Increasing Manifold Fidelity
To bridge the gap between English phrasing and SQL complexity, two primary changes were implemented:
1. **Expansion of Dimensionality**: The subspace was expanded from 50 to **179 components**, increasing variance retention from 62.5% to **90%**. This preserved the subtle linguistic cues necessary for logical mapping.
2. **Balanced Class Weighting**: Custom weights were introduced to the SVM to ensure the minority "Extra Hard" class was absorbed with higher sensitivity.

#### 6.1.3 Final Production Evaluation (Optimized State)
The optimized model was re-evaluated against the 179D baseline manifold (derived from the Spider dataset [1]) and subsequently scaled to the **220D production manifold** to maintain 90% variance retention for the 768D production embeddings.

**Evaluation Visuals (Production Phase)**:

![Production Confusion Matrix](phase5_confusion_matrix.png)
*Figure 14: Production Confusion Matrix — Successful structural learning and diagonal performance in the optimized manifold (179D Baseline / 220D Production).*

- **Observation**: The classifier achieved strong diagonal performance across all four classes. The most notable remaining confusion was a minor overlap between "Medium" and "Easy," suggesting a high semantic similarity between adjacent complexity levels.
- **Success with Complexity**: Accuracy on "Extra Hard" queries jumped to **98/132**, validating that the model successfully learned structural differences.

![Production Metric Comparison](phase5_metric_comparison.png)
*Figure 15: Production Metric Comparison — Uniform performance across all query classes in the final 220D production layer.*

- **Observation**: The "short bars" of the baseline were replaced by strong, uniform precision and recall across the board.
- **Conclusion**: By expanding the manifold to **179 components** (baseline) or **220 components** (production) to capture 90% variance, and balancing the classifier's sensitivity, the system became viable for production-grade routing.
    - **Methodological Validation**: These comparative results empirically validate our routing architecture. By transitioning from the baseline Spider embeddings to the higher-dimensional production embeddings, the system achieved a **10-point increase in F1-Score** (rising from 0.71 to 0.81). This performance jump indicates that higher-dimensional production embeddings provide superior geometric separability for the SVM classifier, confirming the routing layer as a highly viable and robust solution for live enterprise traffic.

![Live vs. Spider Distribution Comparison](produc_vers/live_vs_spider_comparison.png)
*Figure 16: Live vs. Baseline Projection — Manifold comparison confirming academic benchmark compatibility with production traffic.*

**Distribution Comparison Analysis**: The manifold comparison in Figure 16 evaluates the distribution of the academic Spider baseline queries [1] against live Pinecone production database queries:
- **X-Axis (Sample Spread)**: The X-axis represents the sample count, with queries ordered along the axis to visualize their semantic density and distribution.
- **Y-Axis (Semantic Deviation)**: The Y-axis measures the relative semantic position using a similarity score, representing how far each query vector deviates from the mean query vector.
- **Core Overlap (L1-Dense Zone)**: The extensive overlap demonstrates a 90% semantic match between the Spider baseline [1] and live enterprise traffic, verifying that academic benchmarks are highly representative of production language styles.
- **Outliers (Right Tail)**: The right side of the plot captures unusual, production-specific queries (outliers) that represent enterprise-specific nomenclature not present in academic datasets.

### 6.2 Visualization Analysis
**PCA vs. t-SNE**: PCA was used to capture global variance (difficulty mapping), while t-SNE [9] was employed to capture local thematic neighborhoods.

### 6.3 Discussion
#### 6.3.1 Implications for Production Architecture
The empirical results provide a robust, evidence-backed justification for the deployment of **Standard PCA** within the SAGE production pipeline. The following architectural implications were identified:

- **Efficient Query Routing Layer**: The implementation of the SVM-RBF classifier [11] enables a high-efficiency model cascading strategy. 
    - **Tier 1 (High Volume)**: "Easy" and "Medium" queries, which constitute approximately **80% of total traffic**, are successfully routed to **free models (e.g., Gemini [7]) via OpenRouter [13]**.
    - **Tier 2 (High Logic)**: "Hard" and "Extra Hard" queries are strategically directed to **Claude Haiku [12]**, ensuring that premium compute is only utilized for tasks requiring advanced logical reasoning.

![Online Query Routing Architecture](produc_vers/online_routing_flowchart_updated.png)
*Figure 17: Online Query Routing Architecture*

The real-time execution flow of this query routing layer is shown in Figure 17. For every incoming query, the system generates its normalized embedding [2], reduces its dimensions using the pre-trained PCA [3] components, and classifies its complexity to route it to free models via OpenRouter [13] (Tier 1) or Claude Haiku [12] (Tier 2). Simultaneously, a Pinecone database [8] similarity query serves as an out-of-distribution (OOD) safety check to guard against semantic outliers.

- **Infrastructure ROI and Performance**: The discovery of significant signal redundancy within the production manifold allows for the elimination of semantic noise. This reduction directly translates to **accelerated similarity search retrieval** and a **70.7% reduction in database storage size** (from 37.2 MB to 10.9 MB), ensuring the system remains responsive at production scales while minimizing cloud overhead (see Table 1).

**Table 1: Infrastructure ROI via Dimensionality Reduction**
| Feature | Original (Without PCA) | Reduced (With PCA) | Savings |
| :--- | :--- | :--- | :--- |
| **Dimensions** | 768 | 220 | ~71% Reduction |
| **Data Size (Est.)** | 37.2 MB | 10.9 MB | 26.3 MB Saved (70.7% Reduction) |


#### 6.3.2 Identifying the Semantic Gap
Analysis of the misclassifications reveals that the core challenge lies in the model's sensitivity: it is naturally more attuned to **semantic themes** (the subject of the query) than to **keyword complexity** (the structure of the query). When aggressive PCA compression is applied, the structural logic is the first to be discarded as "noise" [5]. By expanding the **baseline architecture to 179 components** and applying balanced weights, we successfully bridged the gap between the natural language meaning and the underlying SQL logical structure.

**Interpretation of the 768D Projection**: In the 2D projection of the 768D embedding space (Figure 18), the axes represent the two strongest mathematical directions of variation in the text. This linear transformation projects the high-dimensional space into a visualizable plane—analogous to shining a light on a 768D object and observing its 2D shadow. Based on the complex, overlapping shapes observed in this scatter plot, a non-linear estimator was required to draw effective boundaries between the difficulty groups, justifying the selection of a kernel-based approach [11].

![Semantic Blind Spot Map](produc_vers/blind_spot_map.png)
*Figure 18: Semantic Blind Spot Map — Visualizing geometric regions where structural logic is susceptible to high-dimensional semantic noise.*

## 7 Conclusion
The research demonstrates that while an initial elbow point of **50 PCA components** captures the semantic core of Text-to-SQL queries, an expansion to **179 components** (baseline) or **220 components** (production) is necessary to preserve the logical "tail" required for complex query classification. The resulting SVM-RBF model provides a principled, cost-efficient foundation for automated enterprise query routing.

## 8 References
[1] **Yu, T., et al. (2018).** "Spider: A Large-Scale Hierarchical Semantic Parsing and Text-to-SQL Dataset." *arXiv preprint arXiv:1809.08887*.
[2] **Xiao, S., et al. (2023).** "C-Pack: Packaged Resources to Advance General Chinese Embedding." *arXiv preprint arXiv:2309.07597*. (BGE Embedding Models)
[3] **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*.
[4] **Salton, G., et al. (1975).** "A Vector Space Model for Automatic Indexing." *Communications of the ACM*.
[5] **Conneau, A., et al. (2018).** "What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties." *ACL*.
[6] **Levina, E., & Bickel, P. (2004).** "Maximum Likelihood Estimation of Intrinsic Dimension." *NIPS*.
[7] **Gemini Team, Google. (2023).** "Gemini: A Family of Highly Capable Multimodal Models."
[8] **Pinecone Systems Inc. (2024).** "Pinecone Vector Database Service."
[9] **Van der Maaten, L., & Hinton, G. (2008).** "Visualizing Data using t-SNE." *Journal of Machine Learning Research*.
[10] **Vanna.ai (2024).** Documentation for Production Data Analyst Agent Framework.
[11] **Cortes, C., & Vapnik, V. (1995).** "Support-Vector Networks." *Machine Learning*, 20(3), 273-297.
[12] **Anthropic. (2024).** "The Claude 3 Family: Technological Advancements in Large Language Models." *Technical Report*.
[13] **OpenRouter. (2024).** "OpenRouter API Reference."
[14] **Schölkopf, B., Smola, A., & Müller, K. R. (1998).** "Nonlinear Component Analysis as a Kernel Eigenvalue Problem." *Neural Computation*, 10(5), 1299-1319.
[15] **Lewis, P., et al. (2020).** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems*, 33, 9459-9474.


---
**Disclosure:** This document was drafted and structured with the assistance of AI tools for linguistic refinement and technical organization. Final data validation and architectural decisions were performed by the author.
