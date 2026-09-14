"""
Copyright (c) 2026 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""


from reader import Reader
import emotion_embedder
import semantic_embedder
import processed_comment
from user import User

emotion_model = emotion_embedder.EmotionEmbedder()
semantic_model = semantic_embedder.SemanticEmbedder()
user_instances = {}  # Dictionary to hold User instances keyed by username

file_reader = Reader("data/reddit_comments.zst")
sampled_data = file_reader.sample_users_with_comments( n_users = 5000, window_start=1609459200,  # Jan 1, 2021
                                                     window_end=1640995200,    # Jan 1, 2022
                                                     min_comments=5, max_comments_per_user=50)
for author, comments in sampled_data.items():
    user_instance = User(username=author)
    for comment in comments:
        # Generate emotion embedding
        emotion_vector = emotion_model.embed_emotion(comment)
        # Generate semantic embedding
        semantic_vector = semantic_model.embed_semantic(comment)
        # Here you can store or process the embeddings as needed
        processed_comment_instance = processed_comment.ProcessedComment(
            comment=comment,
            emotion_embedding=emotion_vector,
            semantic_embedding=semantic_vector
        )
        user_instance.add_comment(processed_comment_instance)

    user_instances[author] = user_instance  # Store the User instance keyed by username     

def merge_all_comments_chronologically(users_dict):
    all_comments = []
    for username, user_obj in users_dict.items():
        all_comments.extend(user_obj.processed_comments)

    all_comments.sort(key=lambda c: c.comment.timestamp)
    return all_comments
