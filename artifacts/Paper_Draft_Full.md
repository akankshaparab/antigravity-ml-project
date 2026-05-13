# SAGE: Cost-Efficient Semantic Query Routing and Optimization for Complex Text-to-SQL Systems

## Abstract
The efficacy of Text-to-SQL systems in Retrieval-Augmented Generation (RAG) architectures depends heavily on the geometric structure of high-dimensional query embedding spaces. This paper investigates the structural properties of these spaces, comparing the academic Spider benchmark (384D) with a production-scale environment (768D) from a Data Analyst agent from vanna.ai. Using Principal Component Analysis (PCA), we characterize the intrinsic dimensionality of the embedding manifold to identify regions of signal redundancy. This geometric assessment informs the development of a Query Routing Layer employing Support Vector Machines (SVM). By mapping cluster cohesion and identifying geometric dispersion among complex query types, this study demonstrates a principled approach to dimensionality reduction and automated query classification. The findings provide a technical framework for optimizing retrieval latency and infrastructure costs by strategically directing queries to Large Language Models based on their latent geometric characteristics, thereby improving the operational efficiency of enterprise Text-to-SQL pipelines.

**Keywords:** Retrieval-Augmented Generation (RAG), Text-to-SQL Systems, Dimensionality Reduction, Principal Component Analysis (PCA), Support Vector Machines (SVM), Query Routing Layer, Embedding Manifold Analysis

## 1 Introduction
In the era of Large Language Models (LLMs), the SAGE (Semantic Automated Geometric Evaluation) platform addresses the critical need for cost-efficient query processing. Enterprise Text-to-SQL systems often face a trade-off between the high accuracy of expensive models and the low latency of smaller models. This paper presents a query routing architecture that uses latent geometric features to optimize this trade-off.

## 2 Prior Research
### Related Work
Text-to-SQL parsing has evolved from rule-based systems to transformer-based architectures. Recent work focuses on embedding-based retrieval and RAG.
### Literature Review
Current literature highlights the performance gap in "Extra Hard" queries but often overlooks the infrastructure cost of high-dimensional vector storage.

## 3 Dataset
### 3.1 Data Description
- **Spider Benchmark**: A cross-domain semantic parsing dataset. Publicly available at [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider). Used as the baseline for academic performance.
- **Production Dataset (vanna.ai)**: A production-scale dataset derived from a Data Analyst agent, representing real-world enterprise query complexity.

### 3.2 Data Statistics
The dataset comprises 11,435 records. The class distributions for both environments are shown below:
- **Spider Distribution**: ![Spider Class Distribution](../spider_class_distribution.png)
- **Production Distribution**: ![Pinecone Distribution](../produc_vers/pinecone_distribution.png)

### 3.3 EDA and preprocessing
- **Spider Preprocessing**: Textual queries were converted into **384-dimensional embeddings** to establish the baseline manifold.
- **Production Preprocessing**: Data was initially converted to **768-dimensional embeddings**.
- **Heuristic Rebalancing**: An initial analysis revealed that the labeling heuristic was biased ("Medium was catching too much" and "Extra Hard was catching too little"). This was rebalanced to ensure data quality.

## 4 Methodology
### 4.1 Dimensionality Reduction
Using PCA, we identified a **220-component elbow point** in the production dataset. This reduction preserves **90% of the variance** while achieving a **70.7% reduction** in storage requirements.
### 4.2 Query Routing and Classification
- **Kernel Comparison**: We performed a parallel comparison of Linear, RBF, and Polynomial kernels. The **RBF kernel** was selected for its ability to resolve the linguistic overlap found in the baseline.
- **Cost-Sensitive Learning**: To resolve the "performance floor" for Extra Hard queries, we implemented **adjusted class weights**, penalizing misclassifications of complex queries more heavily.

## 5 Experiments
### 5.1 Experimental Configuration
Testing was conducted on the optimized 220D manifold.
### 5.2 Hyperparameter Settings
SVM parameters were optimized via grid search on the RBF kernel candidates.
### 5.3 Evaluation Metrics
The evaluation phase documented the evolution of the **Baseline Architecture** (`ml_project`), specifically capturing the performance leap after the heuristic rebalancing and class-weight optimization.

*   **F1, Precision, and Recall (Baseline Optimization)**: The "Before" metrics represent the initial benchmark state where linguistic overlap and heuristic bias suppressed performance. The "After" metrics demonstrate the baseline after adjusting class weights and rebalancing difficulty labels, which stabilized the model before its deployment to production.
*   **The "Extra Hard" Pivot**: This optimization occurred during the baseline phase. By specifically targeting the **Extra Hard** class with adjusted weights, we transformed it from a failure state into a high-recall category. This improved logic was the foundation for the final production architecture.
*   **Accuracy Optimization**: Global accuracy was significantly improved within the baseline by fixing the "Medium-heavy" heuristic. This separate pointer tracks the transition from a biased labeling system to a high-precision classification logic.
*   **Baseline Evolution Visuals (Before vs. After Optimization)**:
    *   **Confusion Matrix Evolution**:
        *   *Initial Baseline (Before)*: ![Initial Baseline](../pre_vers_confu_matr.png)
        *   *Optimized Baseline (After)*: ![Optimized Baseline](../phase5_confusion_matrix.png)
    *   **Metric Comparison Evolution**:
        *   *Initial Baseline (Before)*: ![Baseline Metrics Before](../pre_vers_metric_comp.png)
        *   *Optimized Baseline (After)*: ![Baseline Metrics After](../phase5_metric_comparison.png)



## 6 Results and Discussion
### 6.1 Benchmark Results
The production model achieved an **F1-score of 0.81**, a significant improvement over the baseline's failure on Extra Hard queries.
### 6.2 Visualization Analysis
- **Scree Plot**: ![PCA Scree Plot](../variance_analysis_plot.png)
- **Blind Spot Map**: ![Geometric Blind Spots](../produc_vers/blind_spot_map.png)
### 6.3 Discussion
The results demonstrate that geometric dispersion in the embedding space can be resolved via targeted augmentation and class-weighted SVMs.

## 7 Conclusion
SAGE provides a scalable framework for enterprise Text-to-SQL systems, delivering high accuracy (0.81 F1) with significant infrastructure efficiency (70.7% saving).

## References
[1] Yu, T., et al. (2018). Spider: A Large-Scale Hierarchical Semantic Parsing and Text-to-SQL Dataset. Yale University.
