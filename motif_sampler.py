#!/usr/bin/env python

"""
Finds motifs in a subset of input sequences.

Prints PWM and ids of selected genes to standard output.

Usage:
  find_motifs.py [options] <infile> <outfile> <ks> <Ns>
  find_motifs.py -h | --help

Options:
  --inflection <float>   Inflection point [default: 0.05]
  --burn-iters <int>     Number of iterations before inflection [default: 1000]
  --stop-iters <int>     Iterations without improvement before stopping [default: 1000]
  --restarts <int>       Number of restarts [default: 20]
  --times <int>          Number of motifs to find [default: 1]
  -v --verbose           Print progress to STDERR
  -h --help              Show this screen

"""

import sys
import math
import time

import numpy as np

from docopt import docopt

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from _motif_sampler import make_profile, all_kmer_scores, score_string
from _motif_sampler import choose_index, choose_index_selected
from _motif_sampler import choose_index_selected_unweighted
from _motif_sampler import make_probs, invert_probs, softmax_probs



def n_kmers(text, k):
    return len(text) - k + 1


alphabet = 'ACGT'
letter_dict = dict((char, i) for i, char in enumerate(alphabet))


def convert_string(s):
    return np.array(list(letter_dict[c] for c in s), dtype=np.uint)


def revert_string(a):
    return ''.join(list(alphabet[i] for i in a))


def profile_random_kmer(text, profile):
    """Choose a kmer from `text` weighted by its probability"""
    k = profile.shape[1]
    scores = all_kmer_scores(text, profile, k)
    idx = choose_index(np.exp(scores))
    return text[idx:idx + k]


def format_profile(profile):
    """Format the most probable motif as a string"""
    motif = profile.argmax(axis=0)
    return revert_string(motif)


def score_state(scores, selected):
    """Summarize scores of selected sequences"""
    return scores[selected].sum()


def log_bernoulli(log_p):
    return math.log(np.random.random()) < log_p


def make_profile_helper(motifs, selected, exclude=-1, alphsize=4):
    return make_profile(motifs, selected.astype(np.uint8),
                        exclude, alphsize)


def _sampler_run(seqs, k, N, inflection, burn_iters, stop_iters, verbose=False):
    """Run a round of sampling"""
    n_seqs = len(seqs)
    motif_indices = list(np.random.randint(0, n_kmers(seq, k))
                         for seq in seqs)
    motifs = np.vstack(list(seq[i:i + k]
                            for i, seq in zip(motif_indices, seqs)))
    selected = np.zeros(n_seqs, dtype=np.bool)
    selected[:N] = True
    np.random.shuffle(selected)
    profile = make_profile_helper(motifs, selected)
    scores = np.array(list(score_string(seq, profile)
                           for seq in seqs), dtype=np.float32)
    best_motif = motifs[selected].copy()
    best_profile = profile
    best_selected = selected.copy()
    best_scores = scores
    best_score = score_state(best_scores, best_selected)

    _iter = 0
    iters_unchanged = 0
    temperature = 1

    # compute alpha so temperature will drop to `inflection` after
    # `burn_iters` iterations
    alpha = np.exp(np.log(inflection / temperature) / burn_iters)

    iters_update = 100

    # don't start counting unchanged iterations until after burn-in
    # period, to ensure optimization does not stop during hot period.
    while min(iters_unchanged, _iter - burn_iters) < stop_iters:
        if verbose:
            if _iter % iters_update == 0:
                sys.stderr.write('  iteration: {}, iterations since best: {},'
                                 ' temperature: {}, score: {}\n'.format(
                                     _iter, iters_unchanged, temperature, best_score))
                sys.stderr.flush()

        # compute and transform probabilities
        not_selected = np.invert(selected)
        _selected = selected.astype(np.uint8)
        _not_selected = not_selected.astype(np.uint8)
        probs = np.exp(scores)
        make_probs(probs, _selected)
        invert_probs(probs, _selected)
        make_probs(probs, _not_selected)
        softmax_probs(probs, _selected, temperature)
        softmax_probs(probs, _not_selected, temperature)

        if verbose:
            if _iter % iters_update == 0:
                if len(np.nonzero(probs[selected])[0]) == 1:
                    print('greedy selected sequences')
                if len(np.nonzero(probs[not_selected])[0]) == 1:
                    print('greedy non-selected sequences')

        # swap out a sequence, maybe
        to_remove = choose_index_selected(probs, _selected)
        to_add = choose_index_selected(probs, _not_selected)
        log_ratio = np.log(probs[to_add]) - np.log(probs[to_remove])
        if log_ratio > 0 or log_bernoulli(log_ratio):
            selected[to_remove] = False
            selected[to_add] = True
            motifs[to_add] = profile_random_kmer(seqs[to_add], profile)
            profile = make_profile_helper(motifs, selected)
        else:
            idx = choose_index_selected_unweighted(selected.astype(np.uint8))
            profile = make_profile_helper(motifs, selected, idx)
            motifs[idx] = profile_random_kmer(seqs[idx], profile)

        scores = np.array(list(score_string(seq, profile) for seq in seqs), dtype=np.float32)
        score = score_state(scores, selected)

        # update best
        if score > best_score:
            best_motif = motifs[selected].copy()
            best_score = score
            best_profile = profile
            best_scores = scores
            best_selected = selected.copy()
            iters_unchanged = 0
        else:
            iters_unchanged += 1
        _iter += 1
        temperature = temperature * alpha

    return best_motif, best_profile, best_scores, best_selected, _iter


def sampler(seqs, k, N, inflection, burn_iters, stop_iters, restarts, verbose=False):
    """Run sampler `restarts` times.

    seqs: iterable of strings
    k: int, length of motif
    iters: number of iterations in each run

    """
    results = []
    for i in range(restarts):
        if verbose:
            sys.stderr.write('Restart {} of {}\n'.format(i + 1, restarts))
            sys.stderr.flush()
        results.append(_sampler_run(seqs, k, N, inflection, burn_iters, stop_iters, verbose))
    return max(results, key=lambda args: score_state(args[2], args[3]))


def find_in_file(infile, outfile, ks, Ns, inflection, burn_iters, stop_iters, restarts, times, verbose=False):
    """Runs finder on sequences in a fasta file"""
    records = list(SeqIO.parse(infile, 'fasta'))
    alphabet = set('ACGT')

    # cannot handle ambiguous nucleotides
    records = list(r for r in records if set(r.seq) == alphabet)
    ids = list(r.id for r in records)
    fwd_seqs = list(r.seq for r in records)
    rcomp_seqs = list(s.reverse_complement() for s in fwd_seqs)
    rcomp_ids = list('{} (rcomp)'.format(i) for i in ids)
    all_seqs = fwd_seqs + rcomp_seqs
    all_ids = ids + rcomp_ids

    del records
    del fwd_seqs
    del rcomp_seqs
    for k in ks:
        for N in Ns:
            seqs = list(str(seq) for seq in all_seqs if len(seq) > k)
            seqs = list(convert_string(s) for s in seqs)
            for i in range(times):
                if verbose:
                    sys.stderr.write('k={}, N={}, run {} of {}\n'.format(k, N, i + 1, times))
                    sys.stderr.flush()

                start = time.time()
                motif, profile, scores, selected, iters = sampler(seqs, k, N, inflection, burn_iters, stop_iters, restarts, verbose)
                stop = time.time()
                runtime = stop - start

                info_file = '_'.join([outfile, str(k), str(N), 'info_{}.txt'.format(i)])
                with open(info_file, 'w') as handle:
                    handle.write('k,N,inflection,burn,stop,restarts,runtime,iters')
                    handle.write('\n')
                    handle.write(','.join(map(str, [k, N, inflection, burn_iters, stop_iters, restarts, runtime, iters])))

                motif_file = '_'.join([outfile, str(k), str(N), 'motif_{}.fasta'.format(i)])
                motif_records = list(SeqRecord(Seq(revert_string(m)), id='', description='') for m in motif)
                SeqIO.write(motif_records, motif_file, 'fasta')

                profile_file = '_'.join([outfile, str(k), str(N), 'profile_{}.txt'.format(i)])
                np.savetxt(profile_file, profile)

                gene_file = '_'.join([outfile, str(k), str(N), 'genes_{}.txt'.format(i)])
                with open(gene_file, 'w') as handle:
                    for idx in np.nonzero(selected)[0]:
                        handle.write(all_ids[idx])
                        handle.write('\n')


if __name__ == "__main__":
    args = docopt(__doc__)
    infile = args["<infile>"]
    outfile = args["<outfile>"]
    ks = list(map(int, args['<ks>'].split(',')))
    Ns = list(map(int, args['<Ns>'].split(',')))
    inflection = float(args['--inflection'])
    burn_iters = int(args['--burn-iters'])
    stop_iters = int(args['--stop-iters'])
    restarts = int(args['--restarts'])
    times = int(args['--times'])
    verbose = args['--verbose']
    find_in_file(infile, outfile, ks, Ns, inflection, burn_iters, stop_iters, restarts, times, verbose)
