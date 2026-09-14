"""
Copyright (c) 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): 2026 Ahmet Sualp Gedikli
"""

from comment import Comment
import comment
from processed_comment import ProcessedComment
import numpy as np

class User:


    def __init__(self, username: str):
        
        self.username = username
        self.processed_comments = []  # List to store comments made by the user
        self.comment_count = 0  # Counter for the number of comments
        self.aggregate_emotion_vector = None  # Placeholder for the user's aggregate emotion vector
        self.aggregate_semantic_vector = None  # Placeholder for the user's aggregate semantic vector
        

    def add_comment(self, processed_comment: ProcessedComment):

        self.processed_comments.append(processed_comment)
        self.comment_count += 1

        # First comment initializes the aggregates.
        if self.comment_count == 1:
            self.aggregate_semantic_vector = processed_comment.semantic_embedding
            self.aggregate_emotion_vector = processed_comment.emotion_embedding
            return

        # Running average update.
        self.aggregate_semantic_vector += ( (
            processed_comment.semantic_embedding - self.aggregate_semantic_vector
        ) / self.comment_count )

        self.aggregate_emotion_vector += ( (
            processed_comment.emotion_embedding - self.aggregate_emotion_vector
        ) / self.comment_count )