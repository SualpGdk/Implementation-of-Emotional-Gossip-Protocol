"""
Copyright (c) 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): 2026 Ahmet Sualp Gedikli
"""

import random
import zstandard
import json
from collections import defaultdict
from comment import Comment


class Reader:
    """
    Reads Reddit comments from a compressed Pushshift dataset, organized
    by user rather than by individual comment.

    Comments are grouped by author on load. Users can then be filtered
    by activity window and sampled without repetition; each sampled
    user's comments come back as a stratified, chronologically sorted
    sub-sample rather than the full history.
    """

    def __init__(self, filePath):
        self.filePath = filePath
        self.file_is_compressed = True
        self.users = {}  
        self.previous_users = set()

    def __decompress(self):
        with open(self.filePath, "rb") as compressed_file:
            dctx = zstandard.ZstdDecompressor()
            decompressed_data = dctx.decompress(compressed_file.read())
            json_string = decompressed_data.decode("utf-8")
            self.file_is_compressed = False

            raw = [
                json.loads(line)
                for line in json_string.splitlines()
                if line.strip()
            ]

        grouped = defaultdict(list)
        for c in raw:
            grouped[c['author']].append(c)

        # sort each user's comments chronologically once, up front
        for author, comments in grouped.items():
            comments.sort(key=lambda c: c['created_utc'])
        self.users = grouped

    def _active_authors(self, window_start=None, window_end=None, min_comments=1):
        """
        Returns authors whose activity falls inside [window_start, window_end]
        (unix timestamps) with at least min_comments comments in that window.
        Pass window_start/window_end=None to skip that bound.
        """
        active = []
        for author, comments in self.users.items():
            in_window = [
                c for c in comments
                if (window_start is None or c['created_utc'] >= window_start)
                and (window_end is None or c['created_utc'] <= window_end)
            ]
            if len(in_window) >= min_comments:
                active.append(author)
        return active

    def sample_user(self, window_start=None, window_end=None, min_comments=1):
        """
        Samples one previously-unsampled user active within the given window.

        Returns
        -------
        str
            The sampled author's username.
        """
        if self.file_is_compressed:
            self.__decompress()

        candidates = [
            a for a in self._active_authors(window_start, window_end, min_comments)
            if a not in self.previous_users
        ]
        if not candidates:
            raise Exception("All eligible users have been sampled.")

        author = random.choice(candidates)
        self.previous_users.add(author)
        return author

    def get_user_comments(self, author, window_start=None, window_end=None,
                           max_comments=None):
        """
        Returns a stratified, chronologically sorted sample of a user's
        comments within the given window.

        If max_comments is None, returns the full (windowed) history.
        Otherwise, comments are sampled evenly across the user's active
        period rather than taken from the head/tail, to preserve coverage
        for the EMA update.
        """
        comments = [
            c for c in self.users[author]
            if (window_start is None or c['created_utc'] >= window_start)
            and (window_end is None or c['created_utc'] <= window_end)
        ]

        if max_comments is not None and len(comments) > max_comments:
            # even stride across the sorted list -> spread across time,
            # not clumped at start/end
            step = len(comments) / max_comments
            indices = [int(i * step) for i in range(max_comments)]
            comments = [comments[i] for i in indices]

        return [
            Comment(
                author=c['author'],
                comment=c['body'],
                timestamp=c['created_utc'],
                comment_id=c['id']
            )
            for c in comments
        ]

    def sample_users_with_comments(self, n_users, window_start=None,
                                    window_end=None, min_comments=1,
                                    max_comments_per_user=None):
        """
        Samples n_users users and returns {author: [Comment, ...]}, each
        list stratified and chronologically sorted.
        """
        result = {}
        for _ in range(n_users):
            author = self.sample_user(window_start, window_end, min_comments)
            result[author] = self.get_user_comments(
                author, window_start, window_end, max_comments_per_user
            )
        return result