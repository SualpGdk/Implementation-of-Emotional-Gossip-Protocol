"""
Copyright (c) 2026 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""


from comments import Comment

class ProcessedComment():
    
    def __init__(self, comment: Comment, semantic_embedding, emotion_embedding):
        """
        Initializes a ProcessedComment instance.

        Parameters
        ----------
        comment : Comment
            The original Comment object containing raw metadata.
        semantic_embedding : numpy.ndarray
            Dense semantic embedding representing the meaning of the comment.
        emotion_embedding : dict[str, float]
            Dictionary mapping each emotion label to its predicted confidence score.
        """
        self.comment = comment
        self.semantic_embedding = semantic_embedding
        self.emotion_embedding = emotion_embedding