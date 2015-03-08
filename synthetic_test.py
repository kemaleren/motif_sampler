import random
from pprint import pprint

import numpy as np

from motif_sampler import sampler, format_profile


def random_string(alphabet, length):
    return ''.join(random.choice(alphabet) for _ in range(length))


def run(repeats=10, inflection=0.05, burn_iters=100, stop_iters=100, restarts=100, verbose=False):
    alphabet = 'ACGT'
    n_strings = 50
    length = 100
    N = 5
    k = 10
    successes = 0
    found_strings = 0
    for r in range(repeats):
        print('replicate {} of {}'.format(r + 1, repeats))
        motif = random_string(alphabet, k)
        seqs = list(random_string(alphabet, length) for _ in range(n_strings))
        for i in range(N):
            seqs[i] = ''.join((motif, seqs[i][len(motif):]))
        profile, score, selected = sampler(seqs, k, N, inflection, burn_iters, stop_iters, restarts, verbose=verbose)
        if format_profile(profile) == motif:
            successes += 1
        if set(np.nonzero(selected)[0]) == set(range(N)):
            found_strings += 1
        if verbose:
            print(motif)
            print(format_profile(profile))
            print(np.nonzero(selected)[0])
    print("found motif {} / {} times".format(successes, repeats))
    print("found strings {} / {} times".format(found_strings, repeats))


if __name__ == "__main__":
    run()
