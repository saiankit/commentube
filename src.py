import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, pipeline
from sklearn.cluster import AgglomerativeClustering
from preprocess import pre_process

file = 'comments.json'
comments_df = pd.read_json(file)

sentences = pre_process(comments_df)
print(sentences)

corpus_embeddings, sentence_embeddings = embedd_sentences(sentences, 'sentence-transformers/all-mpnet-base-v2')

# # Perform kmean clustering
clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.7)
clustering_model.fit(sentence_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(sentences[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i+1)
    print(cluster)
    print("")
    print("\n")

def embedd_sentences(sentences, modelName):
    model = SentenceTransformer(modelName)
    sentence_embeddings = model.encode(sentences)
    # Normalize the embeddings to unit length
    corpus_embeddings = sentence_embeddings /  np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    return corpus_embeddings, sentence_embeddings

