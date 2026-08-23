# Study Guide & Practice Script
**Presentation Title:** Analyzing Query Embedding Spaces in RAG-Based Text-to-SQL Systems Using PCA and SVM  
**Speaker:** Akanksha Parab  
**Presentation Date:** August 24, 2026  

---

## Slide 1: Title Slide & Introduction
* **Title:** Analyzing Query Embedding Spaces in RAG-Based Text-to-SQL Systems Using PCA and SVM
* **Subtitle:** Bridging Mathematics and AI
* **Speaker Info:** Akanksha Parab (TYBSc Mathematics Major, Statistics Minor, St. Xavier's College, Mumbai)

### Spoken Script
"Good afternoon, everyone. Thank you for being here. My name is Akanksha Parab, and I am currently in my final year pursuing my BSc in Mathematics with a minor in Statistics here at St. Xavier’s.

Today, I want to talk to you about something that sits right at the intersection of our daily mathematics lectures and the rapidly evolving world of Artificial Intelligence. Over the summer, I did my internship at an AI services startup, where I focused on a research project titled 'Analyzing Query Embedding Spaces in RAG-Based Text-to-SQL Systems Using PCA and SVM.'

Now, that title might sound like a mouthful of computer science jargon, but at its heart, this research is driven entirely by the mathematical concepts we study in our classrooms. If you have ever sat in a Linear Algebra lecture wondering why we study vector spaces, dimension, or eigenvalues, or in a Real Analysis class wondering about metric spaces and distance, this presentation is for you.

We will see how these exact concepts are used to help database systems understand human language. Specifically, we will look at how we convert English questions into high-dimensional geometric coordinates, use Principal Component Analysis to find the 'true' shape of that data, and use Support Vector Machines to classify query complexity so we can route queries to the most cost-effective AI models.

My goal today is to show you that mathematics is not just abstract theory written on a blackboard; it is the concrete architecture of modern AI. Let's start by looking at the core problem my research addresses."

---

## Slide 2: The Core Problem: Enterprise Query Routing
* **Applied Concept:** Finding an optimal decision boundary to route simple queries to a cheap model (Gemini Flash) and complex queries to an expensive model (Gemini Pro).
* **Math Link:** Constrained optimization.

### Spoken Script
"Now let's delve into the actual problem my summer research addressed. In modern enterprise systems, we often build Text-to-SQL pipelines—systems that take a plain English question from a user, translate it into database SQL code, query the database, and return the answer.

To run this in production, we face a major resource trade-off. Large Language Models, like Gemini Pro, are highly accurate but they are slow and extremely expensive. On the other hand, small models, like Gemini Flash, are fast and nearly free, but they fail when they encounter complex database queries, such as those requiring multiple table joins or nested subqueries.

As mathematicians, we can formalize this trade-off as a constrained optimization problem. We want to find a routing function $R$ that maps each query $q$ to either our small model or our large model. Our mathematical objective is to minimize the expected cost and latency $C(R(q))$ of our pipeline, subject to the constraint that our overall accuracy remains above some target threshold $\alpha$.

The practical challenge is: how do we build this routing function if the input is a string of English words? We cannot perform mathematical operations on text directly. We must first project the text into a geometric space where we can run calculations. This brings us to vector space embeddings."

---

## Slide 3: High-Dimensional Vector Spaces & Text Embeddings
* **Applied Concept:** Mapping text to numerical coordinates using Sentence Transformers.
* **Math Link:** High-dimensional real vector spaces ($\mathbb{R}^{768}$) and Cosine Similarity (inner products).

### Spoken Script
"To build a router that classifies queries, we must first convert human language into a format that computers can run calculations on. In our applied AI system, we use a pre-trained Sentence Transformer model. This is the exact same model running on Sumvec's live production platform, ensuring our research matches real-world traffic.

When a user enters a query, the model acts as a function, mapping the text into a 768-dimensional real vector space, $\mathbb{R}^{768}$. Instead of a 2D coordinate like $(x, y)$, a query becomes a list of 768 coordinates. Each coordinate represents a semantic feature learned by the model during training.

The core mathematical idea is that sentences with similar meanings will point in similar directions in this space. To measure this closeness, we use cosine similarity, which computes the cosine of the angle $\theta$ between two vectors. 

As you can see in the diagram on the right, if we project this space to 2D: Query A, 'Select all employees,' and Query B, 'List the staff members,' point in nearly the same direction with a tiny angle $\theta$ between them, yielding a cosine similarity close to $1$. Meanwhile, an unrelated query like 'What is the capital of France?' points in a completely different direction, making it nearly orthogonal, with a similarity close to $0$."

### Q&A Prep
* **Q: Why 768 dimensions?** 
  * *Answer:* This is a design property of the BAAI/bge embedding model. 768 dimensions is the standard capacity required to capture complex English semantics (like syntax, intent, and relationships) without losing meaning.
* **Q: What is a vector space mathematically?**
  * *Answer:* It is a set of elements (vectors) closed under vector addition and scalar multiplication, satisfying standard axioms. Here, it is $\mathbb{R}^{768}$, meaning every vector has 768 real-numbered components.

---

## Slide 4: Vector Norms & Geometric Normalization
* **Applied Concept:** Similarity-based retrieval.
* **Math Link:** Euclidean ($L_2$) Norm and unit normalization.

### Spoken Script
"In applied AI, we want to match queries based on their semantic direction. However, longer sentences naturally contain more words, which can skew the length, or magnitude, of their vectors. If we don't account for this, the system might think two queries are different simply because one has more words than the other.

To prevent this, we perform unit normalization.

Mathematically, we define the length of a vector using the Euclidean norm, or $L_2$ norm. This is the square root of the sum of the squared coordinates. To isolate the direction, we divide our raw vector $\mathbf{x}$ by its norm, producing a normalized unit vector $\hat{\mathbf{x}}$ with a magnitude of exactly $1.0$.

You can see this projection in the 2D diagram on the right. The dashed circle represents the boundary where the distance from the origin is exactly $1.0$. The red arrow is our raw vector, which extends past this boundary. By dividing by its norm, we scale it down along its path until its tip sits exactly on the circle, resulting in the blue vector $\hat{\mathbf{x}}$.

The great mathematical advantage of this is that when all vectors have a length of $1.0$, the denominator in our cosine similarity formula simplifies to $1 \times 1 = 1$. This means similarity can be calculated using a simple dot product, which is computationally very fast. 

Now let's look at how we validate our query difficulty labels."

### Q&A Prep
* **Q: Why does dividing by the norm yield a length of 1?**
  * *Answer:* The norm has a scaling property: $\|c \mathbf{x}\| = |c| \|\mathbf{x}\|$ for any scalar $c$. Since we scale by $c = \frac{1}{\|\mathbf{x}\|_2}$, the norm of the new vector is $\|\hat{\mathbf{x}}\|_2 = \left\| \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \right\|_2 = \frac{\|\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = 1$.
* **Q: What is a unit hypersphere?**
  * *Answer:* It is the multi-dimensional generalization of a circle (in 2D) or a sphere (in 3D). In $d$ dimensions, it is the set of all points at distance $1.0$ from the origin, denoted as $S^{d-1}$.

---

## Slide 5: Data Distribution & Class Imbalance
* **Applied Concept:** Identifying data distribution profiles across Spider and Sumvec Pinecone datasets.
* **Math Link:** Probability distributions and class imbalance statistics.

### Spoken Script
"After vectorizing our queries, the next step is labeling them by complexity. In our research, we work with two distinct datasets. The first is the academic benchmark dataset, Spider, containing 10,000 query pairs mapped to 384-dimensional vectors. The second is the live production database on Pinecone, which contains over 11,000 queries mapped to 768 dimensions.

To classify them, we use a heuristic parser that scores each query based on its SQL structure—assigning points for keywords like JOINs, GROUP BYs, and nested subqueries. This score groups our queries into four difficulty categories: Easy, Medium, Hard, and Extra Hard.

However, looking at the distribution, we observe a classical statistics problem: class imbalance. In both the academic dataset and live production traffic, Easy and Medium queries dominate the dataset, while 'Extra Hard' queries represent a small minority of only 15%.

For our routing classifier, this imbalance is dangerous. If a machine learning model is trained on imbalanced data, it will naturally try to achieve high accuracy by predicting the majority classes and ignoring the minority 'Extra Hard' queries. Since misrouting an 'Extra Hard' query causes a pipeline failure, we must correct this imbalance later. 

But first, we face a major geometric obstacle. With 768 dimensions in production, how do we handle the vast empty space where our data lies? Let's discuss the curse of dimensionality."

### Q&A Prep
* **Q: Why are Easy/Medium queries so dominant?**
  * *Answer:* In real-world enterprise databases, the majority of day-to-day data analyst requests are simple aggregates or single-table lookups. Complex joins and subqueries are much rarer, creating a natural imbalance in live traffic.
* **Q: How does the model over-predict majority classes?**
  * *Answer:* Standard classifiers minimize overall error rate. If a class is 85% of the data, the model can get 85% accuracy by always predicting that class, without learning anything about the minority class.

---

## Slide 6: The Curse of Dimensionality & Redundancy
* **Applied Concept:** Identifying empty space issues and phrasing redundancy (noise).
* **Math Link:** Exponential volume scaling ($s^d$).

### Spoken Script
"Now, why is working in a 768-dimensional space a problem? It comes down to a fundamental mathematical concept called the Curse of Dimensionality.

As we increase the dimension $d$ of a space, its volume grows exponentially. If we divide a 1-dimensional line segment in half, we get 2 sections. For a 2D square, we get 4 quadrants. For a 3D cube, we get 8 octants. If we scale this to a 768-dimensional hypercube, dividing it along each axis gives us $2^{768}$ sub-regions. 

Because this volume is so massive, our 11,000 query vectors are incredibly isolated from one another. They float in a vast, empty space where distance metrics become unstable—meaning the distance between any two queries starts to look nearly identical.

But here is the key semantic insight: natural language is structured and highly redundant. A query vector doesn't need all 768 coordinates to express its meaning. In fact, we discovered that we can discard 10% of the variance in our dataset because it is mostly 'noise.' This noise represents tiny variations in punctuation, capitalization, or spelling that do not change the query's actual database intent.

By removing this noise, we can contract our space and find a much tighter, lower-dimensional coordinate system. To do this, we project our data onto a new mathematical basis using Principal Component Analysis."

### Q&A Prep
* **Q: What is a "manifold" in this context?**
  * *Answer:* A manifold is a lower-dimensional topological space that locally resembles Euclidean space. In text embeddings, it means that although the vectors are written in a 768D space, they actually lie on a tightly constrained, lower-dimensional curved surface representing grammatical and semantic rules.
* **Q: Why is 10% loss considered "noise"?**
  * *Answer:* When we compress vectors, the first few principal components capture the primary semantic concepts (e.g. keywords, operators). The very last components capture tiny, non-semantic details like spacing or minor punctuation differences. Discarding them does not affect Text-to-SQL logic.

---

## Slide 7: Principal Component Analysis (PCA)
* **Applied Concept:** Compressing the 768D production coordinates into an optimized subset.
* **Math Link:** Change of basis and orthonormal bases.

### Spoken Script
"Now that we know the 768-dimensional space contains redundancy and noise, how do we find a better coordinate system? We use Principal Component Analysis, or PCA.

In linear algebra, we learn that a vector space can be represented using different bases. PCA is simply a change of basis. We are searching for a new orthonormal basis—meaning the basis vectors are all unit length and perpendicular to each other—that aligns with the shape of our data. 

The first new basis vector we choose points in the direction of the maximum variance, which is the direction of the greatest spread of our data points. Each subsequent basis vector we choose is orthogonal to the previous ones and captures the next largest spread.

As you can see in the cumulative variance curve on the right, which is the actual plot generated from our live Pinecone production database, the curve climbs very steeply at first. This is because the first few basis vectors capture the main semantic patterns. The curve then flattens out, meaning additional dimensions are just adding tiny details and noise.

By looking at this curve, we found that we can capture 90% of the total variance using only 220 dimensions instead of the original 768. This is a massive 71% reduction in dimensionality, which translates to a major boost in processing speed and storage savings in production."

### Q&A Prep
* **Q: Why is standard PCA chosen over other methods in production?**
  * *Answer:* Standard PCA is a closed-form linear transformation which makes it extremely fast and computationally reliable for streaming live embeddings.
* **Q: Why did you choose a 90% variance threshold?**
  * *Answer:* Retaining 90% of the variance represents the optimal trade-off point. The remaining 10% was shown to be mostly semantic noise (punctuation, spacing, synonyms) that does not help classify SQL difficulty but takes up 71% of the space.

---

## Slide 8: PCA Variants & The Information Loss Trade-off
* **Applied Concept:** Benchmarking alternative versions of PCA on the 384D Spider dataset.
* **Math Link:** Matrix computational trade-offs (scalability vs. sparsity vs. kernels).

### Spoken Script
"While standard PCA is highly effective, we wanted to benchmark it against other variants to see how they handle memory constraints and data structures in a real-world pipeline. We conducted this benchmark on the 384-dimensional academic Spider dataset. 

First, standard PCA is our baseline. It is very fast and has the highest accuracy, but it requires loading the entire dataset into RAM at once. For massive enterprise scale, this is a memory bottleneck.

To solve this, we tested Incremental PCA. Instead of loading all data, it processes the matrix in small, sequential chunks. We found that Incremental PCA consumes almost no RAM while yielding nearly identical coordinates to standard PCA.

Next, we analyzed Sparse PCA. In standard PCA, every principal component is a linear combination of all 384 original dimensions, which makes them hard to interpret. Sparse PCA solves this by adding an L_1-regularization penalty, or Lasso penalty, on the component weights. This forces many weights to become exactly 0. The resulting principal components are composed of only a few key dimensions, making it highly interpretable for domain experts.

Finally, we evaluated Kernel PCA. Standard PCA is a strictly linear transformation. Kernel PCA maps the data into a higher-dimensional space where curved, non-linear relationships become flat and linear. However, because it requires computing a kernel matrix—which means comparing every single query against every other query—its complexity scales quadratically. Our code had to restrict the run to 2,000 rows, which equates to 4 million comparisons, making it too slow for streaming production environments."

### Q&A Prep
* **Q: Why does Kernel PCA scale quadratically?**
  * *Answer:* Kernel PCA calculates a kernel matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ for all pairs of data points. For $N$ samples, this matrix is size $N \times N$, which takes $O(N^2)$ memory and computation.
* **Q: What is a sparse weight in PCA?**
  * *Answer:* In standard PCA, a component is represented as $c = w_1 x_1 + w_2 x_2 + \dots + w_n x_n$ where all weights $w_i \neq 0$. In Sparse PCA, most $w_i$ are set to $0$ (e.g. $c = 0.8 x_2 + 0.3 x_{15}$), meaning the component is constructed from only two features instead of $384$, which is much easier to explain.

---

## Slide 9: Results: Subspace and Blind Spots
* **Applied Concept:** Finding non-linear blind spots in embedding spaces.
* **Math Link:** Linear transformations and projections (2D shadow projection analogy).

### Spoken Script
"After choosing standard PCA as our dimensionality reduction tool, we wanted to look at the geometry of our embedding space. We project our 768-dimensional coordinates onto a 2D plane so we can inspect the distribution of our four complexity classes.

Mathematically, we are applying a linear transformation to project a high-dimensional space down to two dimensions. The best way to visualize this is through a simple analogy: imagine shining a flashlight on a complex, 768-dimensional object. The plot we see on the right is the 2D shadow cast by that object on a flat wall. 

By examining this shadow, we uncovered crucial geometric insights about our data.

First, if you look at the green points, which represent our 'Easy' queries, they are tightly clustered together. This means simple questions share highly repetitive semantic structures—they 'speak the same language' and lie in a very small pocket of the space.

However, look at the orange points representing 'Extra Hard' queries. They do not form a cluster at all. Instead, they are widely dispersed across the outer boundaries of the space. These outliers are our 'blind spots.' They represent complex, rare SQL queries containing deep nests or unions that look completely different from one another.

To resolve this in production, we cannot just hope the model learns these sparse regions. Instead, we use this geometric map to actively augment our database index with a wider, more diverse set of nested-query examples, filling in these blind spots.

Now, seeing that our classes overlap and have curved boundaries in this 2D shadow, how do we draw boundary lines to separate them? This brings us to SVM classification and the Kernel Trick."

### Q&A Prep
* **Q: Why does a linear projection look like a "shadow"?**
  * *Answer:* Mathematically, a shadow is an orthogonal projection of a 3D coordinate onto a 2D plane (e.g. $(x,y,z) \rightarrow (x,y)$). PCA does the exact same thing by projecting $d$-dimensions onto the principal $k$-dimensional subspace, preserving the maximum possible geometric structure.
* **Q: What is a nested query, and why does it cause geometric dispersion?**
  * *Answer:* A nested query contains a query inside another query (e.g., `SELECT ... WHERE ID IN (SELECT...)`). Because they contain multiple subqueries, their phrasing is highly diverse and complex, which spreads their vectors wide across the embedding space compared to standard single-table queries.

---

## Slide 10: SVM Classification & The Kernel Trick
* **Applied Concept:** Constructing decision boundaries to partition queries.
* **Math Link:** Non-linear decision boundaries and kernels (RBF vs. Poly vs. Linear).

### Spoken Script
"Once we mapped our queries into the 220-dimensional subspace, our next task was building the classifier that routes them. We chose Support Vector Machines, or SVM, because they are mathematically designed to find the optimal boundary line that separates different classes of data.

However, our queries are not perfectly separable by a straight line. If we try to use a standard Linear Kernel—which draws a flat boundary—the model fails, achieving a poor accuracy of only 59.3%. This is because the linguistic overlap between classes is too complex.

To solve this, we use the Kernel Trick. The core idea is that if a dataset is not separable in its current space, we can map it to a higher-dimensional space where the curved boundaries become flat, allowing us to separate them cleanly.

We benchmarked different types of kernels in Phase 4 of our research.

First, the Polynomial Kernel captures curved boundaries, achieving a better accuracy of 71.9%. However, it is mathematically more complex to tune, takes longer to execute at nearly 6 seconds, and proved inconsistent when tested on live production traffic.

Second, we tested the Radial Basis Function, or RBF kernel. RBF achieved our highest accuracy at 73.4% and was the fastest, executing in just 3.8 seconds. The mathematical secret behind RBF is that it uses a 'local' approach. It measures the exponential distance between points, meaning it prioritizes nearby queries. This is perfect for capturing the tight, dense 'pockets' of Easy queries we saw in our subspace projection.

Having selected the RBF SVM as our routing engine, let's look at the final evaluation of our routing layer."

### Q&A Prep
* **Q: What does the RBF kernel do mathematically?**
  * *Answer:* The RBF kernel is defined as $K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|_2^2)$. It computes similarity based on the squared Euclidean distance. If two query vectors $\mathbf{x}$ and $\mathbf{y}$ are close, their distance is small, and the exponential is close to $1$. If they are far, the similarity drops exponentially to $0$.
* **Q: Why is RBF faster than Polynomial?**
  * *Answer:* Polynomial kernels require calculating high-degree dot products $( \mathbf{x}^T \mathbf{y} + c )^d$, which is computationally heavier than the exponential distance subtraction in RBF, especially as the number of data points increases.

---

## Slide 11: Results: Classification Performance
* **Applied Concept:** Accuracy improvement from academic dataset to production.
* **Math Link:** Classifier evaluation metrics (Precision, Recall, F1).

### Spoken Script
"In Phase 5, we evaluate our final classifier's performance to see if it meets our production requirements.

When we benchmarked our RBF SVM classifier on the academic Spider dataset in 384 dimensions, it achieved a respectable F1-score of 0.71. But when we migrated the exact same training pipeline to Sumvec's live 768-dimensional production vectors, our F1-score jumped by 10 points to 0.81.

Let's look at the plot on the right to understand why this happens mathematically.

The X-axis represents our sample counts, and the Y-axis is the relative semantic distance from the mean query. The blue points represent the academic Spider dataset, which are tightly compressed along a narrow band. The orange points represent our live production traffic, which spread out widely.

Because the 768-dimensional space is larger, it allows the vector space embedding function to capture fine-grained semantic distinctions, reducing class overlap. Geometrically, this means the points are more spread out, giving our SVM a wider, cleaner margin to draw its decision boundaries.

This performance validates that our classifier is highly reliable. Let's look at the actual business and infrastructure savings of this architecture in production."

### Q&A Prep
* **Q: What is an F1-score, and why is it preferred over raw accuracy here?**
  * *Answer:* The F1-score is the harmonic mean of Precision and Recall. Since our dataset is imbalanced (Extra Hard queries are only 15%), raw accuracy could be misleading. The F1-score evaluates our accuracy per class, ensuring we aren't misclassifying the critical minority classes.
* **Q: Why does higher dimensionality improve classification?**
  * *Answer:* This is a core property of Support Vector Machines. According to Cover's Theorem, a complex non-linear classification problem in low dimensions is more likely to be linearly separable when projected into a higher-dimensional space.

---

## Slide 12: Implications for Production Architecture (Conclusion)
* **Applied Concept:** Operational choices and cloud infrastructure savings.
* **Math Link:** Cost minimization (linear optimization) and subspace dimensions.

### Spoken Script
"To conclude, let's examine the real-world implications of our research on Sumvec's production architecture.

First, our benchmark results prove that standard PCA is the optimal, evidence-backed choice for Pinecone vector compression. By compressing the database from 768 coordinates down to 220 coordinates, we eliminate 71% of redundant, noisy dimensions.

As you can see in the table on the right, this compression reduces our active database footprint from 37.2 megabytes down to just 10.9 megabytes—saving over 26 megabytes. In cloud computing, this reduces the memory footprint on Pinecone, translating directly to lower infrastructure costs and faster similarity search lookups.

Second, our routing layer is highly effective. The SVM classifier successfully identifies that 80% of our daily query traffic consists of Easy or Medium questions. The router directs these to free models from Openrouter, resolving them at zero inference cost. The remaining 20% of complex queries are directed to Claude Haiku, ensuring maximum accuracy for difficult requests.

This realizes the optimization objective we stated at the very beginning of the presentation: minimizing cost and latency while keeping pipeline accuracy above our target threshold.

Thank you, and I am happy to open the floor to any questions."

### Q&A Prep
* **Q: How does reducing vector dimensions speed up similarity search?**
  * *Answer:* Cosine similarity requires calculating dot products ($\sum_{i=1}^d u_i v_i$). Reducing $d$ from 768 to 220 means the processor performs 548 fewer multiplications per comparison, speeding up search latency.
* **Q: Why Claude Haiku for complex queries?**
  * *Answer:* Claude Haiku is highly accurate for SQL logic (JOINs, subqueries) but has a non-zero API cost. The router ensures we only pay this cost for the 20% of queries that actually need it.
