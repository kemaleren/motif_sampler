#!/usr/bin/env python

"""
Finds motifs in a subset of input sequences, using Gibbs sampling.

Usage:
  find_motifs.py [options] <infile> <k> <N>
  find_motifs.py -h | --help

Options:
  --iters <int>          Number of iterations per run [default: 1000]
  --runs <int>           Number of runs [default: 20]
  -v --verbose           Print progress to STDERR
  -h --help              Show this screen

"""

import sys
import math
import random
from functools import reduce
from operator import add
from collections import Counter

import numpy as np
import numpy.random

from docopt import docopt

from Bio import SeqIO



def n_kmers(text, k):
    return len(text) - k + 1


alphabet = 'ACGT'
letter_dict = dict((char, i) for i, char in enumerate(alphabet))


def convert_string(s):
    return np.array(list(letter_dict[c] for c in s), dtype=np.int)


def revert_string(a):
    return ''.join(list(alphabet[i] for i in a))


def score_kmer(text, idx, profile, k):
    """Score a profile with a profile matrix"""
    return reduce(add, (profile[text[idx + i], i] for i in range(k)), 0)


def profile_random_kmer(text, profile):
    """Choose a kmer from `text` weighted by its probability"""
    k = profile.shape[1]
    scores = np.array(list(score_kmer(text, idx, profile, k)
                           for idx in range(n_kmers(text, k))))
    probs = renorm(np.exp2(scores))
    idx = np.random.choice(np.arange(len(probs)), 1, p=probs)
    return text[idx:idx + k]


def score_string(string, profile):
    """The max score over kmers in `string`."""
    k = profile.shape[1]
    return max(score_kmer(string, idx, profile, k)
               for idx in range(n_kmers(string, k)))


def _profile_column(column):
    column = column.ravel()
    column = np.append(column, np.arange(4))  # pseudocounts
    counts = np.bincount(column)
    result = np.log2(counts / counts.sum())
    return result.reshape((-1, 1))


def make_profile(motifs, selected, exclude=None):
    """Make a profile from a list of strings, with pseudocounts"""
    if exclude is not None:
        selected = selected.copy()
        selected[exclude] = False
    motifs = motifs[selected]
    cols = list(_profile_column(motifs[:, c]) for c in range(motifs.shape[1]))
    result = np.hstack(cols)
    return result


def format_profile(profile):
    """Format the most probable motif as a string"""
    motif = profile.argmax(axis=0)
    return revert_string(motif)


def score_state(scores, selected):
    """Summarize scores of selected sequences"""
    return scores[selected].sum()


def _arghelper(scores, selected, operator):
    """Returns (score, index) for `operator` selected score"""
    return operator((s, i) for i, s in enumerate(scores) if selected[i])


def argmin(scores, selected):
    return _arghelper(scores, selected, min)


def argmax(scores, selected):
    return _arghelper(scores, selected, max)


def log_bernoulli(log_p):
    return math.log2(random.random()) < log_p


def renorm(probs):
    return probs / probs.sum()


def swap_cands(scores, selected):
    included = np.nonzero(selected)[0]
    excluded = np.nonzero(1 - selected)[0]
    included_idx = np.random.choice(included, 1,
                                    p=renorm(np.exp2(scores[included])))[0]
    excluded_idx = np.random.choice(excluded, 1,
                                    p=renorm(np.exp2(scores[excluded])))[0]
    return included_idx, excluded_idx


def _gibbs_round(seqs, k, N, iters, verbose=False):
    """Run a round of Gibbs sampling."""
    n_seqs = len(seqs)
    motif_indices = list(random.choice(range(n_kmers(seq, k))) for seq in seqs)
    motifs = np.vstack(list(seq[i:i + k] for i, seq in zip(motif_indices, seqs)))
    selected = np.zeros(n_seqs, dtype=np.bool)
    selected[:N] = True
    np.random.shuffle(selected)
    profile = make_profile(motifs, selected)
    scores = np.array(list(score_string(seq, profile) for seq in seqs))
    best_profile = profile
    best_selected = selected.copy()
    best_scores = scores
    best_score = score_state(best_scores, best_selected)
    for i in range(iters):
        if verbose:
            if i % 100 == 0:
                sys.stderr.write('  iteration {} of {}\n'.format(i, iters))
                sys.stderr.flush()

        # swap out a sequence, maybe
        to_remove, to_add = swap_cands(scores, selected)
        log_ratio = scores[to_add] - scores[to_remove]
        if log_ratio > 0 or log_bernoulli(log_ratio):
            selected[to_remove] = False
            selected[to_add] = True
            motifs[to_add] = profile_random_kmer(seqs[to_add], profile)
            profile = make_profile(motifs, selected)
        else:
            idx = random.choice(np.nonzero(selected)[0])
            profile = make_profile(motifs, selected, idx)
            motifs[idx] = profile_random_kmer(seqs[idx], profile)

        scores = np.array(list(score_string(seq, profile) for seq in seqs))
        score = score_state(scores, selected)

        # update best
        if score > best_score:
            best_score = score
            best_profile = profile
            best_scores = scores
            best_selected = selected.copy()
    return best_profile, best_scores, best_selected


def gibbs(seqs, k, N, iters, starts, verbose=False):
    """Run Gibbs sampling `starts` times.

    seqs: iterable of strings
    k: int, length of motif
    iters: number of iterations in each run

    """
    results = []
    seqs = list(convert_string(s) for s in seqs)
    for i in range(starts):
        if verbose:
            sys.stderr.write('Starting run {} of {}.\n'.format(i + 1, starts))
            sys.stderr.flush()
        results.append(_gibbs_round(seqs, k, N, iters, verbose))
    return max(results, key=lambda args: score_state(args[1], args[2]))


def find_in_file(infile, k, N, iters, starts, verbose=False):
    """Runs finder on sequences in a fasta file"""
    seqs = SeqIO.parse(infile, 'fasta')
    profile, _, _ = gibbs(seqs, k, N, iters, starts, verbose)
    print(profile)


if __name__ == "__main__":
    args = docopt(__doc__)
    infile = args["<infile>"]
    k = int(args['<k>'])
    N = int(args['<N>'])
    iters = int(args['--iters'])
    runs = int(args['--runs'])
    verbose = args['--verbose']
    find_in_file(infile, k, iters, starts, verbose)
