import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import zipfile

print("Loading model and getting embeddings...")
df = pd.DataFrame({'text': ['This is a sentence.', 'This is another sentence.']})
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist())
print("Embeddings shape:", embeddings.shape)
print("Done!")
# Load larger transformer model
model_bge = SentenceTransformer('BAAI/bge-small-en-v1.5')

def encode_spider_questions(zip_path, model):
    questions = []
    print(f"Reading Spider dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Load questions from both the train and dev files inside the zip
        for filename in ['spider_data/train_spider.json', 'spider_data/train_others.json', 'spider_data/dev.json']:
            if filename in z.namelist():
                with z.open(filename) as f:
                    data = json.load(f)
                    for item in data:
                        if 'question' in item:
                            questions.append(item['question'])
                            
    print(f"Extracted {len(questions)} questions. Encoding now...")
    # Use the model to generate embeddings, enabling progress bar for the large list
    spider_embeddings = model.encode(questions, show_progress_bar=True)
    print("Spider embeddings shape:", spider_embeddings.shape)
    return spider_embeddings

# Define the zip path and call the function
spider_zip_path = r'C:\Users\Dell1234\Downloads\spider_data.zip'
spider_embeddings = encode_spider_questions(spider_zip_path, model_bge)