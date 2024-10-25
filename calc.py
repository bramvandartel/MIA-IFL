import numpy as np


def tpr(conf_matrix):  # Recall
    return conf_matrix[3] / (conf_matrix[3] + conf_matrix[2])


def precision(conf_matrix):
    if conf_matrix[3] + conf_matrix[1] == 0:
        return 0
    return conf_matrix[3] / (conf_matrix[3] + conf_matrix[1])


def fpr(conf_matrix):
    return conf_matrix[1] / (conf_matrix[1] + conf_matrix[0])


def adv(conf_matrix):
    return tpr(conf_matrix) - fpr(conf_matrix)


def avg_adv(conf_matrices):
    for matrix in conf_matrices:
        assert len(matrix) == 4
        print(f"TPR: {tpr(matrix)}, FPR: {fpr(matrix)}, ADV: {adv(matrix)}, Precision: {precision(matrix)}")

    print(f"Avg TPR: {np.mean([tpr(conf_matrix) for conf_matrix in conf_matrices])},"
          f"Avg FPR: {np.mean([fpr(conf_matrix) for conf_matrix in conf_matrices])},"
          f"Avg Adv: {np.mean([adv(conf_matrix) for conf_matrix in conf_matrices])},"
          f"Avg Precision: {np.mean([precision(conf_matrix) for conf_matrix in conf_matrices])}")