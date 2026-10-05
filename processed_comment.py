"""
Copyright (c) 2026 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""


from comments import Comment
import numpy as np  # ADDED (point 4)

class ProcessedComment():
    
    def __init__(self, comment: Comment, semantic_embedding, emotion_embedding, emotion_labels=None):  # CHANGED (point 4): added emotion_labels
        """
        Initializes a ProcessedComment instance.

        Parameters
        ----------
        comment : Comment
            The original Comment object containing raw metadata.
        semantic_embedding : numpy.ndarray
            Dense semantic embedding representing the meaning of the comment.
        emotion_embedding : numpy.ndarray or dict[str, float]
            Emotion scores. CHANGED (point 4): a dict is converted to a fixed-order float32
            vector here, because every consumer (BubbleNode, User, tokens) does vector math on it.
        emotion_labels : list[str], optional
            ADDED (point 4): label order used when emotion_embedding is a dict. Defaults to
            sorted(keys), so pass an explicit list if your classifier has a canonical order.
        """
        self.comment = comment
        self.semantic_embedding = semantic_embedding
        # CHANGED (point 4): was `self.emotion_embedding = emotion_embedding` (could be a dict)
        if isinstance(emotion_embedding, dict):
            labels = list(emotion_labels) if emotion_labels is not None else sorted(emotion_embedding.keys())
            emotion_embedding = [emotion_embedding[label] for label in labels]
        self.emotion_embedding = np.asarray(emotion_embedding, dtype=np.float32)