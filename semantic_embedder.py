"""
Copyright (c) 2026 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""


from sentence_transformers import SentenceTransformer
from comments import Comment


class SemanticEmbedder:
    """
    Generates semantic embeddings for textual comments.

    A SentenceTransformer model is loaded once during initialization
    and reused for all embedding requests to avoid the overhead of
    repeatedly loading the neural network.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the semantic embedding model.

        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model to load.
        """
        self.model = SentenceTransformer(model_name)

    def embed_semantic(self, comment: Comment):
        """
        Generates a semantic embedding for a comment.

        Parameters
        ----------
        comment : Comment
            Comment whose textual content will be embedded.

        Returns
        -------
        numpy.ndarray
            Dense semantic embedding representing the meaning of the
            comment.
        """

        semantic_embedding = self.model.encode(comment.body)

        return semantic_embedding