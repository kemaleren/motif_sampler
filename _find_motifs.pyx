#!python
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cdef double score_kmer(unsigned long[:] text, unsigned int idx,
               double[:, :] profile, unsigned int k) nogil:
    cdef double total = 0
    cdef unsigned int i
    for i in range(k):
        total += profile[text[idx + i], i]
    return total


def all_kmer_scores(unsigned long[:] text, double[:, :] profile, unsigned int k):
    cdef unsigned int n_kmers = text.shape[0] - k + 1
    result = np.empty(n_kmers, dtype=np.dtype("d"))
    cdef double [:] result_view = result
    cdef unsigned int i
    with nogil:
        for i in range(n_kmers):
            result_view[i] = score_kmer(text, i, profile, k)
    return result


def score_string(unsigned long[:] text, double[:, :] profile):
    cdef unsigned int k = profile.shape[1]
    cdef unsigned int n_kmers = text.shape[0] - k + 1
    cdef double result = score_kmer(text, 0, profile, k)
    cdef double cur
    cdef unsigned int i
    with nogil:
        for i in range(n_kmers):
            cur = score_kmer(text, i, profile, k)
            if cur > result:
                result = cur
    return result
