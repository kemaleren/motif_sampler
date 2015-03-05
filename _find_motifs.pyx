#!python
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np


def make_profile(unsigned long[:, :] motifs, np.uint8_t[:] selected,
                 long exclude, unsigned int alphabet_size):
    cdef unsigned int n_motifs = motifs.shape[0]
    cdef unsigned int k = motifs.shape[1]

    counts = np.ones((alphabet_size, k), dtype=np.uint64)
    cdef unsigned long [:, :] counts_view = counts
    result = np.zeros((alphabet_size, k), dtype=np.dtype('d'))
    cdef double [:, :] result_view = result

    cdef long denom = alphabet_size

    cdef unsigned int i
    cdef unsigned int j

    for i in range(n_motifs):
        if (not selected[i]) or (i == exclude):
            continue
        denom += 1
        for j in range(k):
            counts_view[motifs[i, j], j] += 1

    cdef double ddenom = <double> denom
    for i in range(alphabet_size):
        for j in range(k):
            result_view[i, j] = np.log2(<double> (counts_view[i, j]) / ddenom)
    return result


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
