import json

import numpy as np

# import pandas as pd
import scipy.stats as stats

problems = ['DTLZ1', 'DTLZ2', 'DTLZ7']
moeas = ['RVEAa', 'metaRVEAa'] # , 'mopso'

results = {}

for m in moeas:
    results[m] = []

print(results)
for p in problems:
    print(p)
    igds = []
    mean_igds = []
    for m in moeas:
        igd = []
        if m=='RVEAa':
            for n in range(31):
                data = json.load(open("../data/model_performance/{}_{}_exp{}.json".format(m, p, n), 'r'))
                igd1 = data['igd'][0]
                igd.append(igd1)
        else:
            for n in range(31):
                data = json.load(open("../data/model_performance/{}_{}_exp{}.json".format(m, p, n), 'r'))
                igd1 = data['igd'][0][0]
                igd.append(igd1)
        mean_igd = np.mean(igd)
        std_igd = np.std(igd)
        igds.append(igd)
        mean_igds.append(mean_igd)
        results[m].append(f"{mean_igd:.4f} ({std_igd:.4f})")
        print("{} igd: {:.4f} ({})".format(m, mean_igd, std_igd))
    igd = np.array(igds)
    mean_igd = np.array(mean_igds)
    print(mean_igds)
    max_idx = np.argmax(mean_igd)
    print(max_idx)
    for i in range(len(moeas)):
        s, p = stats.ranksums(igd[max_idx], igd[i])
        print("p value of alg {}: {} {}".format(i, p, p>0.05))
    print('\n')


# df = pd.DataFrame(results, index=problems)
# excel_path = 'igd_Values.xlsx'
# df.to_excel(excel_path)
# print(f"Data saved to {excel_path}")