import random
from pprint import pprint

import numpy as np

from find_motifs import gibbs, format_profile


def random_string(alphabet, length):
    return ''.join(random.choice(alphabet) for _ in range(length))


def run():
    alphabet = 'ACGT'
    length = 100
    k = 12
    N = 25
    n_strings = 30
    iters = 500
    starts = 10
    motif = random_string(alphabet, k)
    seqs = list(random_string('ACGT', length) for _ in range(n_strings))
    for i in range(N):
        seqs[i] = ''.join((motif, seqs[i][len(motif):]))
    profile, score, selected = gibbs(seqs, k, N, iters, starts, verbose=True)
    print(motif)
    print(format_profile(profile))
    print(np.nonzero(selected)[0])


if __name__ == "__main__":
    run()
