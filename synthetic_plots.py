import os
import random
import time
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as pp

from motif_sampler import sampler, format_profile


replicates = 10


experiments = [
    {'name': 'motif length',
     'n_strings': 50,
     'length': 1000,
     'N': 10,
     'k': [5, 10, 15, 20, 25],
     'iters': 1000,
     'starts': 10,
 },
    {'name': 'gene length',
     'n_strings': 50,
     'length': [100, 250, 500, 750, 1000],
     'N': 10,
     'k': 10,
     'iters': 1000,
     'starts': 10,
 },
    {'name': 'genes with motifs',
     'n_strings': 50,
     'length': 1000,
     'N': [5, 10, 15, 20, 25],
     'k': 10,
     'iters': 1000,
     'starts': 10,
 },
    {'name': 'sampling iterations',
     'n_strings': 50,
     'length': 1000,
     'N': 10,
     'k': 10,
     'iters': [100, 250, 500, 750, 1000],
     'starts': 10,
 },
    {'name': 'sampling restarts',
     'n_strings': 50,
     'length': 1000,
     'N': 10,
     'k': 10,
     'iters': 1000,
     'starts': [1, 5, 10, 15, 20],
 },
]


def random_string(alphabet, length):
    return ''.join(random.choice(alphabet) for _ in range(length))


def edit_distance(a, b):
    n = len(a) + 1
    m = len(b) + 1
    mat = np.zeros((n, m), dtype=np.int)
    mat[:, 0] = np.arange(n)
    mat[0, :] = np.arange(m)
    for i in range(1, n):
        for j in range(1, m):
            chra = a[i - 1]
            chrb = b[j - 1]
            mat[i, j] = min(mat[i - 1, j - 1] + (0 if chra == chrb else 1),
                            mat[i - 1, j] + 1,
                            mat[i, j - 1] + 1)
    return mat[n - 1, m - 1]


def run(name, n_strings, length, k, N,
        iters=500, starts=10):
    params = locals()
    alphabet = 'ACGT'
    records = []
    for r in range(replicates):
        motif = random_string(alphabet, k)
        seqs = list(random_string('ACGT', length) for _ in range(n_strings))
        for i in range(N):
            # TODO: insert in random places
            # TODO: mutate motif
            seqs[i] = ''.join((motif, seqs[i][len(motif):]))
        start = time.time()
        profile, score, selected = sampler(seqs, k, N, iters, starts)
        stop = time.time()
        runtime = (stop - start)
        motif_dist = edit_distance(motif, format_profile(profile)) / k
        found_strings = len(set(np.nonzero(selected)[0]) & set(range(N))) / N
        entropy = -sum(logp * np.exp(logp) for row in profile for logp in row) / k
        record = {'runtime': runtime,
                  'motif_distance': motif_dist,
                  'found_strings': found_strings,
                  'motif_entropy': entropy
        }
        record.update(params)
        records.append(record)
    return records


def experiment_variable(experiment):
    for k, v in experiment.items():
        if isinstance(v, list):
            return k
    raise Exception()


def experiment_iter(experiment):
    key = experiment_variable(experiment)
    for val in experiment[key]:
        result = experiment.copy()
        result[key] = val
        yield result


def run_all(outdir):
    all_records = []
    outfile = os.path.join(outdir, 'results.csv')
    for experiment in experiments:
        print(experiment['name'])
        xkey = experiment_variable(experiment)
        for params in experiment_iter(experiment):
            print(params[xkey])
            records = run(**params)
            all_records.extend(records)

            # save all data so far
            df = pd.DataFrame(all_records)
            df.to_csv(outfile)

        # plots for this experiment
        for ykey in ['runtime', 'motif_distance', 'found_strings', 'motif_entropy']:
            filename = '{}_{}.png'.format(experiment['name'], ykey)
            samples = df[df['name'] == experiment['name']]
            pp.figure()
            samples.boxplot(column=ykey, by=xkey)
            pp.suptitle('')
            pp.xlabel(experiment['name'])
            pp.ylabel(ykey)
            pp.title('')
            pp.savefig(os.path.join(outdir, filename))
        # runtime
        # motif distance
        # found strings
        # motif entropy



if __name__ == "__main__":
    run_all(sys.argv[1])
