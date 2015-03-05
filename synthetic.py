import random
from pprint import pprint

import numpy as np

from find_motifs import gibbs, format_profile


def random_string(alphabet, length):
    return ''.join(random.choice(alphabet) for _ in range(length))


def run(iters=1000, starts=1):
    alphabet = 'ACGT'
    n_strings = 20
    N = 5
    length = 100
    k = 10
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
