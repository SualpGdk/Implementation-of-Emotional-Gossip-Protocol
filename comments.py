"""
Copyright (c) 2026

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): Ahmet Sualp Gedikli
"""

from dataclasses import dataclass


@dataclass
class Comment:
    """
    Represents a single Reddit comment read from the dataset.

    This class stores only the raw metadata extracted from the dataset.
    Additional information such as semantic embeddings and emotion
    vectors is generated later during preprocessing.

    Attributes
    ----------
    author : str
        Username of the comment's author.

    comment : str
        Raw textual content of the comment.

    timestamp : str
        UTC timestamp indicating when the comment was created.

    comment_id : str
        Unique identifier assigned to the comment.
    """

    author: str
    body: str
    timestamp: str
    comment_id: str