#!/usr/bin/env python
"""

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

import numpy
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
    """Score a profile with a profile matrix

    >>> str(round(score('aa', [{'a': 0.7, 'g': 0.3}, {'a': 0.7, 'g': 0.3}]), 2))
    '0.49'

    """
    return reduce(add, (math.log2(p[l]) for l, p in zip(kmer, profile)), 0)


def score_string(string, profile):
    k = len(profile)
    return max(score_kmer(kmer, profile) for kmer in kmers(string, k))


def profile_random_kmer(text, profile):
    """Choose a kmer from `text` weighted by its probability"""
    k = len(profile)
    _kmers = list(kmers(text, k))
    probs = list(math.pow(2, score_kmer(kmer, profile)) for kmer in _kmers)
    s = sum(probs)
    probs = list(p / s for p in probs)
    result = numpy.random.choice(_kmers, 1, p=probs)
    return result[0]


def select(iterable, selected):
    return list(elt for elt, s in zip(iterable, selected) if s)


def invert(selected):
    return list(not s for s in selected)


def _profile(letters):
    counter = Counter(letters)
    counter.update("ACGT")
    total = len(letters) + 4
    result = {letter: (count / total)
              for letter, count in counter.items()}
    return {letter: result[letter] for letter in 'ACGT'}


def make_profile(motifs, selected, exclude=None):
    """Make a profile from a list of strings, with pseudocounts"""
    motifs = select(motifs, selected)
    if exclude is not None:
        motifs = motifs[:exclude] + motifs[exclude + 1:]
    return list(_profile(col) for col in zip(*motifs))


def score_state(scores, selected):
    return min(select(scores, selected))


def argmin(scores, selected):
    return min((s, i) for i, s in enumerate(scores) if selected[i])


def argmax(scores, selected):
    return max((s, i) for i, s in enumerate(scores) if selected[i])


def _gibbs(seqs, k, N, iters, verbose=False):
    n_seqs = len(seqs)
    motifs = list(random.choice(list(kmers(seq, k))) for seq in seqs)
    selected = [True] * N + [False] * (n_seqs - N)
    random.shuffle(selected)
    profile = make_profile(motifs, selected)
    scores = list(score_string(seq, profile) for seq in seqs)
    best_profile = profile
    best_selected = selected
    best_scores = scores
    for i in range(iters):
        if verbose:
            if i % 100 == 0:
                sys.stderr.write('  iteration {} of {}\n'.format(i, iters))
                sys.stderr.flush()

        # swap out a sequence, if possible
        incl_score, included = argmin(scores, selected)
        excl_score, excluded = argmax(scores, invert(selected))
        if excl_score > incl_score:
            # TODO: do this probabilistically
            assert(selected[included])
            assert(not selected[excluded])
            selected[included] = False
            selected[excluded] = True
            idx = excluded
            profile = make_profile(motifs, selected, idx)
            motifs[idx] = profile_random_kmer(seqs[idx], profile)
        else:
            idx = random.choice(list(i for i, s in enumerate(selected) if i))
            profile = make_profile(motifs, selected, idx)
            motifs[idx] = profile_random_kmer(seqs[idx], profile)

        scores = list(score_string(seq, profile) for seq in seqs)

        # update best
        if score_state(scores, selected) > score_state(best_scores, best_selected):
            best_profile = profile
            best_scores = scores
            best_selected = selected
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
            sys.stderr.write('Starting run {} of {}.\n'.format(i, starts))
            sys.stderr.flush()
        results.append(_gibbs(seqs, k, N, iters, verbose))
    return max(results, key=lambda args: score_state(args[1], args[2]))


def find_in_file(infile, k, N, iters, starts, verbose=False):
    seqs = SeqIO.parse(infile, 'fasta')
    profile, _, _ = gibbs(seqs, k, N, iters, starts, verbose)
    print(profile)


def format_profile(profile):
    return ''.join(list(max(c, key=c.get) for c in profile))


if __name__ == "__main__":
    args = docopt(__doc__)
    infile = args["<infile>"]
    k = int(args['<k>'])
    N = int(args['<N>'])
    iters = int(args['--iters'])
    runs = int(args['--runs'])
    verbose = args['--verbose']
    find_in_file(infile, k, iters, starts, verbose)
