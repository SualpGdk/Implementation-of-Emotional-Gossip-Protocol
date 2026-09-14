"""
Copyright (c) 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): 2026 Ahmet Sualp Gedikli
"""

import io
import json
import random
import zstandard

from comments import Comment


class Reader:

    def __init__(self, filePath):
        self.filePath = filePath
        self.file_is_compressed = True
        self.users = {}
        self.previous_users = set()

    def sample_users_with_comments(
        self,
        n_users,
        min_comments=1,
        max_comments_per_user=None
    ):
        # -------------------------
        # PASS 1
        # -------------------------
        print("Starting pass 1...")

        eligible_authors = self.first_pass(
            min_comments=min_comments,
            filepath=self.filePath
        )

        print(
            f"Found {len(eligible_authors):,} eligible authors."
        )

        if len(eligible_authors) < n_users:
            raise ValueError(
                f"Only {len(eligible_authors)} eligible users found, "
                f"but {n_users} were requested."
            )

        # -------------------------
        # PASS 2
        # -------------------------
        print(f"Starting pass 2 for {n_users:,} users...")

        return self.second_pass(
            eligible_authors,
            n_users,
            filepath=self.filePath,
            max_comments_per_user=max_comments_per_user
        )

    def first_pass(self, min_comments=1, filepath=None):

        comment_counts = {}
        lines_processed = 0
        invalid_records = 0

        dctx = zstandard.ZstdDecompressor()

        with open(filepath, "rb") as compressed_file:

            with dctx.stream_reader(compressed_file) as decompressed_stream:

                text_stream = io.TextIOWrapper(
                    decompressed_stream,
                    encoding="utf-8"
                )

                for line in text_stream:

                    if not line.strip():
                        continue

                    lines_processed += 1

                    try:
                        json_representation = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_records += 1
                        if lines_processed >= 10_000_000:
                            break
                        continue

                    author = json_representation.get("author")

                    if author is None:
                        invalid_records += 1
                        if lines_processed >= 10_000_000:
                            break
                        continue

                    comment_counts[author] = (
                        comment_counts.get(author, 0) + 1
                    )

                    if lines_processed % 1_000_000 == 0:
                        print(
                            f"Pass 1: processed "
                            f"{lines_processed:,} records"
                        )

                    if lines_processed >= 10_000_000:
                        break

        eligible_authors = [
            author
            for author, count in comment_counts.items()
            if count >= min_comments
            and author not in self.previous_users
        ]

        print(
            f"Pass 1 complete. "
            f"Processed {lines_processed:,} records."
        )

        if invalid_records:
            print(
                f"Skipped {invalid_records:,} invalid records."
            )

        return eligible_authors

    def second_pass(
        self,
        eligible_authors,
        n_users,
        filepath=None,
        max_comments_per_user=None
    ):

        sampled_authors = random.sample(
            eligible_authors,
            n_users
        )

        sampled_author_set = set(sampled_authors)

        result = {
            author: []
            for author in sampled_authors
        }

        lines_processed = 0
        invalid_records = 0
        comments_collected = 0

        dctx = zstandard.ZstdDecompressor()

        with open(filepath, "rb") as compressed_file:

            with dctx.stream_reader(compressed_file) as decompressed_stream:

                text_stream = io.TextIOWrapper(
                    decompressed_stream,
                    encoding="utf-8"
                )

                for line in text_stream:

                    if not line.strip():
                        continue

                    lines_processed += 1

                    try:
                        json_representation = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_records += 1
                        if lines_processed >= 10_000_000:
                            break
                        continue

                    author = json_representation.get("author")

                    if author not in sampled_author_set:
                        if lines_processed >= 10_000_000:
                            break
                        continue

                    body = json_representation.get("body")
                    timestamp = json_representation.get("created_utc")
                    comment_id = json_representation.get("id")

                    # Skip malformed records
                    if (
                        body is None
                        or timestamp is None
                        or comment_id is None
                    ):
                        invalid_records += 1
                        if lines_processed >= 10_000_000:
                            break
                        continue

                    comment = Comment(
                        author=author,
                        body=body,
                        timestamp=timestamp,
                        comment_id=comment_id
                    )

                    result[author].append(comment)
                    comments_collected += 1

                    if lines_processed % 1_000_000 == 0:
                        print(
                            f"Pass 2: processed "
                            f"{lines_processed:,} records | "
                            f"collected {comments_collected:,} comments"
                        )

                    if lines_processed >= 10_000_000:
                        break

        # -------------------------
        # Sort + stratified sampling
        # -------------------------

        for author in result:

            comments = result[author]

            comments.sort(
                key=lambda c: c.timestamp
            )

            if (
                max_comments_per_user is not None
                and len(comments) > max_comments_per_user
            ):

                step = (
                    len(comments)
                    / max_comments_per_user
                )

                indices = [
                    int(i * step)
                    for i in range(max_comments_per_user)
                ]

                result[author] = [
                    comments[i]
                    for i in indices
                ]

        self.previous_users.update(sampled_authors)

        print(
            f"Pass 2 complete. "
            f"Collected {comments_collected:,} comments."
        )

        if invalid_records:
            print(
                f"Skipped {invalid_records:,} invalid records."
            )

        return result
"""
Copyright (c) 

This source code is licensed under the MIT license found in the 
LICENSE file in the root directory of this source tree.

Author(s): 2026 Ahmet Sualp Gedikli
"""
"""

import io
import random
import zstandard
import json
from comments import Comment


class Reader:

    def __init__(self, filePath):
        self.filePath = filePath
        self.file_is_compressed = True
        self.users = {}
        self.previous_users = set()

    def _extract_comment_data(self, record: dict):
    
        author = record.get("author")
        if not author or author in ("[deleted]", "[removed]"):
            return None

        body = record.get("body")
        if not body or body in ("[deleted]", "[removed]"):
            return None

        timestamp = record.get("created_utc")
        comment_id = record.get("id")

        if timestamp is None or not comment_id:
            return None

        return author, str(body), str(timestamp), str(comment_id)

    def sample_users_with_comments(
        self,
        n_users,
        min_comments=1,
        max_comments_per_user=None
    ):
        # -------------------------
        # PASS 1
        # -------------------------
        print("Starting pass 1...")

        eligible_authors = self.first_pass(
            min_comments=min_comments,
            filepath=self.filePath
        )

        print(
            f"Found {len(eligible_authors):,} eligible authors."
        )

        if len(eligible_authors) < n_users:
            raise ValueError(
                f"Only {len(eligible_authors)} eligible users found, "
                f"but {n_users} were requested."
            )

        # -------------------------
        # PASS 2
        # -------------------------
        print(f"Starting pass 2 for {n_users:,} users...")

        return self.second_pass(
            eligible_authors,
            n_users,
            filepath=self.filePath,
            max_comments_per_user=max_comments_per_user
        )

    def first_pass(self, min_comments=1, filepath=None):
        comment_counts = {}
        lines_processed = 0
        invalid_records = 0

        dctx = zstandard.ZstdDecompressor()

        with open(filepath, "rb") as compressed_file:
            with dctx.stream_reader(compressed_file) as decompressed_stream:
                text_stream = io.TextIOWrapper(
                    decompressed_stream,
                    encoding="utf-8",
                    errors="replace"
                )

                for line in text_stream:
                    if not line.strip():
                        continue

                    lines_processed += 1

                    try:
                        json_representation = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_records += 1
                        continue

                    extracted = self._extract_comment_data(json_representation)
                    if extracted is None:
                        invalid_records += 1
                        continue

                    author = extracted[0]
                    comment_counts[author] = (
                        comment_counts.get(author, 0) + 1
                    )

                    if lines_processed % 2_000_000 == 0:
                        print(
                            f"Pass 1: processed "
                            f"{lines_processed:,} records"
                        )

        eligible_authors = [
            author
            for author, count in comment_counts.items()
            if count >= min_comments
            and author not in self.previous_users
        ]

        print(
            f"Pass 1 complete. "
            f"Processed {lines_processed:,} records."
        )

        if invalid_records:
            print(
                f"Skipped {invalid_records:,} invalid/deleted records."
            )

        return eligible_authors

    def second_pass(
        self,
        eligible_authors,
        n_users,
        filepath=None,
        max_comments_per_user=None
    ):
        sampled_authors = random.sample(
            eligible_authors,
            n_users
        )

        sampled_author_set = set(sampled_authors)

        result = {
            author: []
            for author in sampled_authors
        }

        lines_processed = 0
        invalid_records = 0
        comments_collected = 0

        dctx = zstandard.ZstdDecompressor()

        with open(filepath, "rb") as compressed_file:
            with dctx.stream_reader(compressed_file) as decompressed_stream:
                text_stream = io.TextIOWrapper(
                    decompressed_stream,
                    encoding="utf-8",
                    errors="replace"
                )

                for line in text_stream:
                    if not line.strip():
                        continue

                    lines_processed += 1

                    try:
                        json_representation = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_records += 1
                        continue

                    extracted = self._extract_comment_data(json_representation)
                    if extracted is None:
                        invalid_records += 1
                        continue

                    author, body, timestamp, comment_id = extracted

                    if author not in sampled_author_set:
                        continue

                    comment = Comment(
                        author=author,
                        body=body,
                        timestamp=timestamp,
                        comment_id=comment_id
                    )

                    result[author].append(comment)
                    comments_collected += 1

                    if lines_processed % 2_000_000 == 0:
                        print(
                            f"Pass 2: processed "
                            f"{lines_processed:,} records | "
                            f"collected {comments_collected:,} comments"
                        )

        # -------------------------
        # Sort + stratified sampling
        # -------------------------
        for author in result:
            comments = result[author]

            # Parse timestamps safely for sorting
            comments.sort(
                key=lambda c: float(c.timestamp) if c.timestamp.replace('.', '', 1).isdigit() else 0
            )

            if (
                max_comments_per_user is not None
                and len(comments) > max_comments_per_user
            ):
                step = (
                    len(comments)
                    / max_comments_per_user
                )

                indices = [
                    int(i * step)
                    for i in range(max_comments_per_user)
                ]

                result[author] = [
                    comments[i]
                    for i in indices
                ]

        self.previous_users.update(sampled_authors)

        print(
            f"Pass 2 complete. "
            f"Collected {comments_collected:,} comments."
        )

        if invalid_records:
            print(
                f"Skipped {invalid_records:,} invalid/deleted records."
            )

        return result"""