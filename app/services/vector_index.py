from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SearchHit:
    chunk_idx: int
    score: float


class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.vectors = self._normalize(vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchHit]:
        q = self._normalize(query_vector.reshape(1, -1))[0]
        scores = self.vectors @ q

        top_indices = np.argsort(-scores)[:top_k]
        return [SearchHit(chunk_idx=int(i), score=float(scores[i])) for i in top_indices]

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
