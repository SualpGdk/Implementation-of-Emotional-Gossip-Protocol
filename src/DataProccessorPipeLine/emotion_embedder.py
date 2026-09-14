"""
Copyright (c) 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): 2026 Ahmet Sualp Gedikli
"""

from transformers import pipeline
from comment import Comment
import numpy as np

class EmotionEmbedder:
    """
    Generates emotion probability distributions for textual comments.

    The underlying transformer model is trained on the GoEmotions
    dataset and produces confidence scores for all supported emotion
    categories.
    """

    def __init__(self):
        """
        Loads the GoEmotions classifier.

        The model is loaded once during construction and reused for all
        subsequent emotion extraction requests.
        """

        self.classifier = pipeline(
            model="SamLowe/roberta-base-go_emotions",
            top_k=None
        )

    def embed_emotion(self, comment: Comment):
        """
        Predicts the emotional distribution of a comment.

        Parameters
        ----------
        comment : Comment
            Comment whose emotional content will be analyzed.

        Returns
        -------
        numpy.ndarray
            Dense vector of confidence scores for each emotion category.
        """

        raw_emotion_embedding = self.classifier(comment.comment)

        emotion_embedding = {
            item["label"]: item["score"]
            for item in raw_emotion_embedding
        }
        emotion_embedding_vector = np.array(list(emotion_embedding.values()))

        return emotion_embedding_vector