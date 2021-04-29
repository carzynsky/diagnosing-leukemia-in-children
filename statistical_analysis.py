import cross_validation
import load_data
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate

def run_statistical_analysis(X, y):

    # prepare data
    clfs = {
        'MLP_15F_32N_0M':  cross_validation.run_crossvalid(X, y, 15, 32, 0),
        'MLP_15F_32N_09M':  cross_validation.run_crossvalid(X, y, 15, 32, 0.9),
        'MLP_15F_64N_0M':  cross_validation.run_crossvalid(X, y, 15, 64, 0),
        'MLP_15F_624_09M':  cross_validation.run_crossvalid(X, y, 15, 64, 0.9),
        'MLP_15F_256N_0M':  cross_validation.run_crossvalid(X, y, 15, 256, 0),
        'MLP_15F_256N_09M':  cross_validation.run_crossvalid(X, y, 15, 256, 0.9)
    }

    folds = []

    # data frame
    for id, name in enumerate(clfs):
        scores = clfs[name]
        folds.append(scores)

    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i,j], p_value[i,j] = ttest_ind(folds[i], folds[j])

    print('t-statistic:\n', t_statistic, '\n\np-value:\n', p_value)


    df_t = pd.DataFrame(t_statistic, index=list(clfs.keys()), columns=list(clfs.keys()))
    # print(df_t)

    df_p = pd.DataFrame(p_value, index=list(clfs.keys()), columns=list(clfs.keys()))
    # print(df_p)

    # advantage
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1

    df_a = pd.DataFrame(advantage, index=list(clfs.keys()), columns=list(clfs.keys()))
    # print('\n\nAdvantage:\n', df_a)

    # statistical significance (alpha = 0.05)
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1

    df_significance = pd.DataFrame(significance, index=list(clfs.keys()), columns=list(clfs.keys()))
    # print('\n\nStatistical signifance (alpha = 0.05):\n', df_significance)

    # statistical significantly better
    stat_better = significance * advantage
    df_better = pd.DataFrame(stat_better, index=list(clfs.keys()), columns=list(clfs.keys()))
    # print('\n\nStatistical significantly better:\n', df_better)


    with pd.ExcelWriter('Data/statistical_analysis.xls') as writer:
        df_t.to_excel(writer,'t_statistic')
        df_p.to_excel(writer,'p_value')
        df_a.to_excel(writer,'advantage')
        df_significance.to_excel(writer,'significance')
        df_better.to_excel(writer, 'stat_better')
        writer.save()

 