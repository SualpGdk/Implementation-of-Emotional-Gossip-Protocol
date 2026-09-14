"""
Copyright (c) 2026 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""

from comments import Comment
import comments
from processed_comment import ProcessedComment
import numpy as np

class User:
    """
    Represents a Reddit user and their aggregated comment embeddings.

    This class stores the processed comments made by a single user and keeps
    running averages for their semantic and emotion embeddings. These aggregates
    can later be used for user-level analysis or similarity comparisons.

    Attributes
    ----------
    username : str
        Username of the user.

    processed_comments : list[ProcessedComment]
        Collection of processed comments associated with the user.

    comment_count : int
        Number of comments stored for the user.

    aggregate_emotion_vector : numpy.ndarray or None
        Running average of the emotion embeddings for all comments.

    aggregate_semantic_vector : numpy.ndarray or None
        Running average of the semantic embeddings for all comments.
    """

    def __init__(self, username: str):
        """
        Initializes a User instance.

        Parameters
        ----------
        username : str
            Username of the user to represent.
        """
        self.username = username
        self.processed_comments = []  # List to store comments made by the user
        self.comment_count = 0  # Counter for the number of comments
        self.aggregate_emotion_vector = None  # Placeholder for the user's aggregate emotion vector
        self.aggregate_semantic_vector = None  # Placeholder for the user's aggregate semantic vector

    def add_comment(self, processed_comment: ProcessedComment):
        """
        Adds a processed comment to the user and updates the running aggregates.

        Parameters
        ----------
        processed_comment : ProcessedComment
            The processed comment containing semantic and emotion embeddings.
        """
        self.processed_comments.append(processed_comment)
        self.comment_count += 1

        # First comment initializes the aggregates.
        if self.comment_count == 1:
            self.aggregate_semantic_vector = processed_comment.semantic_embedding
            self.aggregate_emotion_vector = processed_comment.emotion_embedding
            return

        # Running average update.
        self.aggregate_semantic_vector += ((
            processed_comment.semantic_embedding - self.aggregate_semantic_vector
        ) / self.comment_count)

        self.aggregate_emotion_vector += ((
            processed_comment.emotion_embedding - self.aggregate_emotion_vector
        ) / self.comment_count)