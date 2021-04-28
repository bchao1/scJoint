import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



def plot_knn_experiment():
    k = [16, 32, 64, 128, 256, 300, 350, 400, 450, 512, 600, 700, 800, 900, 1024]
    acc = [
        0.9381765696, 
        0.9379360115, 
        0.9420254992,
        0.9453933125,
        0.9444310801,
        0.941544383,
        0.9198941544,
        0.8138080346,
        0.8109213375,
        0.8075535242,
        0.7914361318,
        0.6867933606,
        0.6665864806,
        0.6519124369,
        0.6223237912
    ]
    plt.figure()
    sns.set_theme()
    ax = sns.lineplot(x = k, y = acc)
    ax.set(xlabel = "knn neighbors", ylabel = "knn accuracy")
    plt.savefig("output/knn_exp.png")

def plot_p_experiment():
    p = 0.1 * np.arange(1, 10)
    acc = [
        0.9331248497,
        0.9328842916,
        0.9321626173,
        0.9312003849,
        0.9350493144,
        0.9381765696,
        0.9374548954,
        0.9374548954,
        0.9374548954
    ]
    plt.figure()
    sns.set_theme()
    ax = sns.lineplot(x = p, y = acc)
    ax.set(xlabel = "fraction to align", ylabel = "knn accuracy")
    plt.savefig("output/p_exp.png")

def plot_batch_experiment():
    b = np.arange(5, 12)
    acc = [
        0.9215780611,
        0.9352898725,
        0.9295164782,
        0.9374548954,
        0.931440943,
        0.9283136878,
        0.8828482078
    ]
    plt.figure()
    sns.set_theme()
    ax = sns.lineplot(x = b, y = acc)
    ax.set(xlabel = "batch size", ylabel = "knn accuracy")
    plt.savefig("output/batch_exp.png")

def encoding_loss_ablation():
    baseline = 0.9374548954
    ablation = [
        0.8655280250180418,
        0.31056050036083716,
        0.9136396439740198
    ]
    for acc in ablation:
        drop = (baseline - acc) / baseline * 100
        print(drop)
        
if __name__ == "__main__":
    plot_knn_experiment()
    plot_p_experiment()
    plot_batch_experiment()
    encoding_loss_ablation()