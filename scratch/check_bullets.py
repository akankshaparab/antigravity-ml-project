import re

with open(r"c:\Users\Dell1234\.gemini\antigravity\scratch\ml_project\paper_revised_draft.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

headings = [
    "Context", "the problem", "objective", "baseline architecture", "production architecture",
    "data matrix formatting", "geometric retrieval theory", "difficulty classifier logic",
    "heuristic rebalancing", "geometric validation", "complementary roles of PCA and SVM",
    "benchmarking methodology", "component rationale", "scree plot insights",
    "kernel comparison and selection", "weight optimization", "embedding engine",
    "data split", "identification of the elbow zone", "statistical imbalance",
    "information loss in the tail", "evaluation visuals (initial phase)",
    "evaluation visuals (production phase)", "distribution comparison analysis",
    "PCA vs t-SNE", "interpretation of the 768D projection"
]

found_bullet = False
for i, line in enumerate(lines):
    for h in headings:
        if re.search(r'\b' + re.escape(h) + r'\b', line, re.IGNORECASE):
            if re.match(r'^\s*[-*+]\s+', line):
                print(f"Match found with bullet on line {i+1}: {line.strip()}")
                found_bullet = True

if not found_bullet:
    print("Verification passed: None of the target headings have bullets beside them.")
