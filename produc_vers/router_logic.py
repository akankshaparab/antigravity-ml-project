import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class QueryRouter:
    """
    Production-ready Query Routing Layer.
    Routes queries to GEMINI FLASH or GEMINI PRO based on predicted complexity.
    """
    
    def __init__(self, artifact_path='router_artifacts.pkl', model_name='BAAI/bge-base-en-v1.5'):
        self.artifact_path = artifact_path
        self.embed_model = SentenceTransformer(model_name)
        self.pca = None
        self.clf = None
        self.is_loaded = False
        
        if os.path.exists(artifact_path):
            self.load_artifacts()

    def load_artifacts(self):
        """Load the trained PCA and SVM models."""
        try:
            with open(self.artifact_path, 'rb') as f:
                artifacts = pickle.load(f)
                self.pca = artifacts['pca']
                self.clf = artifacts['clf']
                self.is_loaded = True
            print(f"Successfully loaded routing artifacts from {self.artifact_path}")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            self.is_loaded = False

    def train_and_save(self, embeddings_path='spider_768_embeddings.npz'):
        """Train the PCA and SVM model on the provided embeddings and save."""
        print(f"Training routing model using {embeddings_path}...")
        data = np.load(embeddings_path)
        X = data['X']
        y = data['y']
        
        # PCA for 90% variance
        pca = PCA(n_components=0.90)
        X_pca = pca.fit_transform(X)
        
        # SVM Classifier
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
        clf.fit(X_pca, y)
        
        # Save Artifacts
        artifacts = {
            'pca': pca,
            'clf': clf,
            'classes': clf.classes_
        }
        
        with open(self.artifact_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        self.pca = pca
        self.clf = clf
        self.is_loaded = True
        print(f"Model trained and saved to {self.artifact_path}")

    def route(self, query, pinecone_results=None):
        """
        Main routing logic.
        Returns the decision and metadata.
        """
        if not self.is_loaded:
            return {"error": "Router artifacts not loaded. Call train_and_save() first."}

        # 1. Generate Embedding
        vec = self.embed_model.encode([query], normalize_embeddings=True)
        
        # 2. Dimensionality Reduction
        vec_pca = self.pca.transform(vec)
        
        # 3. Predict Complexity
        complexity = self.clf.predict(vec_pca)[0]
        probs = self.clf.predict_proba(vec_pca)[0]
        max_prob = np.max(probs)
        
        # 4. Routing Decision
        # Threshold: Easy/Medium -> Flash, Hard/Extra Hard -> Pro
        if complexity in ["Easy", "Medium"]:
            target = "GEMINI FLASH"
        else:
            target = "GEMINI PRO"
            
        # 5. Hybrid Guardrail (If Pinecone results provided)
        # If the nearest neighbor in Pinecone is very far, default to PRO for safety
        guardrail_triggered = False
        if pinecone_results and len(pinecone_results.get('matches', [])) > 0:
            top_score = pinecone_results['matches'][0]['score']
            if top_score < 0.60:  # OOD Threshold
                target = "GEMINI PRO"
                guardrail_triggered = True

        return {
            "query": query,
            "decision": target,
            "complexity": complexity,
            "confidence": round(float(max_prob), 4),
            "guardrail_triggered": guardrail_triggered
        }

if __name__ == "__main__":
    # Self-test / Deployment script
    router = QueryRouter()
    
    # If artifacts don't exist, train them
    if not router.is_loaded:
        router.train_and_save('spider_768_embeddings.npz')
    
    # Test queries
    test_queries = [
        "What is the average age of all users?",
        "Show me the total revenue grouped by month and region for products that had more than 100 sales in 2023, excluding returns"
    ]
    
    print("\n--- Routing Tests ---")
    for q in test_queries:
        result = router.route(q)
        print(f"Query: {q}")
        print(f"Result: {result['decision']} ({result['complexity']} | Conf: {result['confidence']})\n")
