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


def kmers(text, k):
    """Generates all `k`-length strings in `text`.

    >>> list(kmers("abcd", 2))
    ['ab', 'bc', 'cd']

    """
    for i in range(len(text) - k + 1):
        yield text[i:i + k]


def score_kmer(kmer, profile):
    """Score a profile with a profile matrix"""
    return reduce(add, (p[l] for l, p in zip(kmer, profile)), 0)


def score_string(string, profile):
    """The max score over kmers in `string`."""
    k = len(profile)
    return max(score_kmer(kmer, profile) for kmer in kmers(string, k))


def profile_random_kmer(text, profile):
    """Choose a kmer from `text` weighted by its probability"""
    k = len(profile)
    _kmers = list(kmers(text, k))
    scores = np.array(list(score_kmer(kmer, profile) for kmer in _kmers))
    probs = np.exp2(scores)
    s = sum(probs)
    probs = list(p / s for p in probs)
    result = np.random.choice(_kmers, 1, p=probs)
    return result[0]


def select(iterable, selected):
    return list(elt for elt, s in zip(iterable, selected) if s)


def invert(selected):
    return 1 - selected


def _profile(letters):
    alphabet = 'ACGT'  # TODO: do not assume alphabet
    counter = Counter(letters)
    counter.update(alphabet)  # pseudocounts
    total = len(letters) + 4
    result = {letter: math.log2(count / total)
              for letter, count in counter.items()}
    return {letter: result[letter] for letter in alphabet}


def make_profile(motifs, selected, exclude=None):
    """Make a profile from a list of strings, with pseudocounts"""
    motifs = select(motifs, selected)
    if exclude is not None:
        motifs = motifs[:exclude] + motifs[exclude + 1:]
    return list(_profile(col) for col in zip(*motifs))


def format_profile(profile):
    """Format the most probable motif as a string"""
    return ''.join(list(max(c, key=c.get) for c in profile))


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
    excluded = np.nonzero(invert(selected))[0]
    included_idx = np.random.choice(included, 1,
                                    p=renorm(np.exp2(scores[included])))[0]
    excluded_idx = np.random.choice(excluded, 1,
                                    p=renorm(np.exp2(scores[excluded])))[0]
    return included_idx, excluded_idx


def _gibbs_round(seqs, k, N, iters, verbose=False):
    """Run a round of Gibbs sampling."""
    n_seqs = len(seqs)
    motifs = list(random.choice(list(kmers(seq, k))) for seq in seqs)
    selected = np.zeros(n_seqs, dtype=np.bool)
    selected[:N] = True
    np.random.shuffle(selected)
    profile = make_profile(motifs, selected)
    scores = np.array(list(score_string(seq, profile) for seq in seqs))
    best_profile = profile
    best_selected = selected
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
