"""
Copyright (c) 2026 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""

import os
import pickle
import hdbscan
import numpy as np
import umap
from reader import Reader
from emotion_embedder import EmotionEmbedderr
from semantic_embedder import SemanticEmbedder
from processed_comment import ProcessedComment
from user import User

CACHE_FILE = "cached_processed_comments.pkl"

print("Starting the main script...")

# ---------------------------------------------------------
# 1. Load from Disk or Run Sampling & Embedding Pipeline
# ---------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"Loading cached comments and embeddings from '{CACHE_FILE}'...")
    with open(CACHE_FILE, "rb") as f:
        all_comments_non_chronologically = pickle.load(f)
    print(f"Loaded {len(all_comments_non_chronologically):,} processed comments from cache.")
else:
    print("No cache found. Running sampling and embedding generation...")
    
    emotion_model = EmotionEmbedderr()
    semantic_model = SemanticEmbedder()
    all_comments_non_chronologically = []

    file_reader = Reader("RC_2019-04.zst")
    sampled_data = file_reader.sample_users_with_comments(
        n_users=900,
        min_comments=5,
        max_comments_per_user=50
    )
    print("User sampling completed.")
    total_comments = sum(len(comments) for comments in sampled_data.values())
    print(f"Total comments to process: {total_comments}")

    processed = 0

    for author, comments in sampled_data.items():
        for comment_instance in comments:
            emotion_vector = emotion_model.embed_emotion(comment_instance)
            semantic_vector = semantic_model.embed_semantic(comment_instance)

            processed_comment_instance = ProcessedComment(
                comment=comment_instance,
                emotion_embedding=emotion_vector,
                semantic_embedding=semantic_vector
            )

            all_comments_non_chronologically.append(processed_comment_instance)
            processed += 1

            if processed % 1000 == 0:
                print(
                    f"Processed {processed:,}/{total_comments:,} "
                    f"({processed / total_comments:.1%})"
                )

    print(f"Saving processed comments to '{CACHE_FILE}' for future runs...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_comments_non_chronologically, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Cache saved successfully.")

if not all_comments_non_chronologically:
    raise ValueError("No comments were collected or loaded.")

# ---------------------------------------------------------
# 2. Reconstruct User Instances
# ---------------------------------------------------------
# Reconstruct user dictionary so author instances are fresh and correctly mapped
unique_authors = {pc.comment.author for pc in all_comments_non_chronologically}
user_instances = {author: User(username=author) for author in unique_authors}

# ---------------------------------------------------------
# 3. Chronological Sorting and Initial Slice
# ---------------------------------------------------------
all_comments_non_chronologically.sort(key=lambda x: x.comment.timestamp)
all_comments_chronologically = all_comments_non_chronologically

ten_percent_index = max(1, len(all_comments_chronologically) // 10)
initial_clustering_comments = all_comments_chronologically[:ten_percent_index]
remaining_comments = all_comments_chronologically[ten_percent_index:]

for processed_comment in initial_clustering_comments:
    user_instance = user_instances[processed_comment.comment.author]
    user_instance.add_comment(processed_comment)

# Filter users who received comments in the initial phase
users_for_initial_clustering = [
    user for user in user_instances.values() if user.comment_count > 0
]

if not users_for_initial_clustering:
    raise ValueError("No users found with comments for initial clustering.")

embeddings = np.array([
    user.aggregate_semantic_vector for user in users_for_initial_clustering
], dtype=np.float32)

print(f"Clustering {len(users_for_initial_clustering)} users...")

# ---------------------------------------------------------
# 4. Dimension Reduction & Dynamic Clustering
# ---------------------------------------------------------
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    metric='cosine',
    min_dist=0.0,
    random_state=42
)
reduced_embeddings = reducer.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=2,
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(reduced_embeddings)

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = int(np.sum(labels == -1))

print("Clustering completed.")
print(f"Dynamic clusters found: {num_clusters}")
print(f"Users classified as noise (-1): {noise_count} / {len(labels)}")