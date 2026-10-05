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

    def __init__(self, username: str, rolling_alpha: float = 0.2):  # CHANGED: added rolling_alpha
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

        # ADDED: recent-behaviour state (EMA). The cumulative aggregates above anchor a user to
        # their whole history; the rolling vectors track what they talk like *now* and drive migration.
        self.rolling_alpha = rolling_alpha
        self.rolling_semantic_vector = None
        self.rolling_emotion_vector = None

        # ADDED: migration state
        self.last_migration_step: int = -10**9
        self.migration_streak: int = 0
        self.migration_target: int | None = None
        self.transition_probabilities: dict[int, float] = {}

    def add_comment(self, processed_comment: ProcessedComment):
        """
        Adds a processed comment to the user and updates the running aggregates.

        Parameters
        ----------
        processed_comment : ProcessedComment
            The processed comment containing semantic and emotion embeddings.
        """
        # ADDED (point 1): private float32 copies of the comment's embeddings.
        sem = np.array(processed_comment.semantic_embedding, dtype=np.float32)
        emo = np.array(processed_comment.emotion_embedding, dtype=np.float32)

        self.processed_comments.append(processed_comment)
        self.comment_count += 1

        # First comment initializes the aggregates.
        if self.comment_count == 1:
            # CHANGED (point 1): was `= processed_comment.semantic_embedding` (same object), so the
            # in-place `+=` below silently mutated the first comment's embedding. Now we copy.
            self.aggregate_semantic_vector = sem.copy()
            self.aggregate_emotion_vector = emo.copy()
            # ADDED: rolling vectors start at the first comment
            self.rolling_semantic_vector = sem.copy()
            self.rolling_emotion_vector = emo.copy()
            return

        # Running average update.
        self.aggregate_semantic_vector += ((
            sem - self.aggregate_semantic_vector  # CHANGED (point 1): uses the private copy
        ) / self.comment_count)

        self.aggregate_emotion_vector += ((
            emo - self.aggregate_emotion_vector  # CHANGED (point 1): uses the private copy
        ) / self.comment_count)

        # ADDED: exponential moving average of recent behaviour
        a = self.rolling_alpha
        self.rolling_semantic_vector = (1 - a) * self.rolling_semantic_vector + a * sem
        self.rolling_emotion_vector = (1 - a) * self.rolling_emotion_vector + a * emo

    # ADDED: used by the migration similarity (Hellinger needs a proper histogram)
    def rolling_emotion_histogram(self) -> np.ndarray:
        """Rolling emotion vector as a histogram (non-negative, sums to 1)."""
        v = np.clip(self.rolling_emotion_vector, 0, None)
        s = v.sum()
        return v / s if s > 0 else v
