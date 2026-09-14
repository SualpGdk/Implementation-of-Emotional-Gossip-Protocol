"""
Copyright (c) 2026

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""

from transformers import pipeline
from comments import Comment
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

    """def embed_emotion(self, comment: Comment):

        raw_emotion_embedding = self.classifier(comment.body)

        emotion_embedding = {
            item["label"]: item["score"]
            for item in raw_emotion_embedding
        }
        emotion_embedding_vector = np.array(list(emotion_embedding.values()))

        return emotion_embedding_vector"""

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
        # Take the first element [0] from the pipeline output
        raw_emotion_embedding = self.classifier(comment.body)[0]

        emotion_embedding = {
            item["label"]: item["score"]
            for item in raw_emotion_embedding
        }
        emotion_embedding_vector = np.array(list(emotion_embedding.values()))

        return emotion_embedding_vector

"""
Copyright (c) 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): 2026 Ahmet Sualp Gedikli
"""

from transformers import pipeline
from comments import Comment
import numpy as np


class EmotionEmbedderr:
    """
    Generates emotion probability distributions for textual comments.

    The underlying transformer model is trained on the GoEmotions
    dataset and produces confidence scores for all supported emotion
    categories.
    """

    def __init__(self):
        """
        Loads the GoEmotions classifier.
        """
        self.classifier = pipeline(
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
            max_length=512
        )

    def embed_emotion(self, comment: Comment):
        """
        Predicts the emotional distribution of a comment.
        """
        raw_emotion_embedding = self.classifier(
            comment.body,
            truncation=True,
            max_length=512
        )[0]

        emotion_embedding = {
            item["label"]: item["score"]
            for item in raw_emotion_embedding
        }
        emotion_embedding_vector = np.array(list(emotion_embedding.values()))

        return emotion_embedding_vector