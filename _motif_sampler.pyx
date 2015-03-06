#!python
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np


def choose_index(double[:] weights):
    """Choose an index, weighted by `weights`."""
    cdef double total = 0
    cdef unsigned int i
    for i in range(weights.shape[0]):
        total += weights[i]

    cdef double x = total * np.random.random()
    cdef double w = 0
    for i in range(weights.shape[0]):
        w += weights[i]
        if w > x:
            return i
    assert(False)


def invert_selected(double[:] weights, np.uint8_t[:] selected):
    cdef double total = 0
    cdef unsigned int i
    result = np.empty_like(weights)
    cdef double [:] result_view = result
    for i in range(weights.shape[0]):
        if selected[i]:
            total += weights[i]
    for i in range(weights.shape[0]):
        if selected[i]:
            result_view[i] = 1 - (weights[i] / total)
    return result


def choose_index_selected(double[:] weights, np.uint8_t[:] selected):
    """Choose a selected index, weighted by `weights`."""
    cdef double total = 0
    cdef unsigned int i
    for i in range(weights.shape[0]):
        if selected[i]:
            total += weights[i]

    cdef double x = total * np.random.random()
    cdef double w = 0
    for i in range(weights.shape[0]):
        if selected[i]:
            w += weights[i]
            if w > x:
                return i
    assert(False)


def choose_index_selected_unweighted(np.uint8_t[:] selected):
    """Choose a selected index"""
    cdef double total = 0
    cdef unsigned int i
    for i in range(selected.shape[0]):
        if selected[i]:
            total += 1

    cdef double x = total * np.random.random()
    cdef double w = 0
    for i in range(selected.shape[0]):
        if selected[i]:
            w += 1
            if w > x:
                return i
    assert(False)


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
