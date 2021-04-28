import process_db
import h5py
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(1)

if __name__ == "__main__":
    rna_embeddings = np.loadtxt('./output/citeseq_control_rna_embeddings.txt')
    atac_embeddings = np.loadtxt('./output/asapseq_control_atac_embeddings.txt')
    print(rna_embeddings.shape)
    print(atac_embeddings.shape)
    embeddings =  np.concatenate((rna_embeddings, atac_embeddings))
    print(embeddings.shape)
    tsne_results = TSNE(perplexity=30, n_iter = 1000).fit_transform(embeddings)
    df = pd.DataFrame()
    df['tSNE1'] = tsne_results[:,0]
    df['tSNE2'] = tsne_results[:,1]

    rna_labels = np.loadtxt('./data/citeseq_control_cellTypes.txt')
    atac_labels = np.loadtxt('./data/asapseq_control_cellTypes.txt')
    atac_predictions = np.loadtxt('./output/asapseq_control_atac_knn_predictions.txt')
    labels =  np.concatenate((rna_labels, atac_predictions))
    label_to_idx = pd.read_csv('./data/label_to_idx.txt', sep = '\t', header = None)
    label_to_idx.shape
    label_dic = []
    for i in range(label_to_idx.shape[0]):
        label_dic = np.append(label_dic, label_to_idx[0][i][:-2])
    common_label_cnt = 0
    correct = 0
    for i in range(len(atac_labels)):
        if atac_labels[i] >= 0:
            common_label_cnt += 1
            if atac_labels[i] == atac_predictions[i]:
                correct += 1
    acc = correct/common_label_cnt
    print(acc)
    
    data_label = np.array(["CITE-seq", "ASAP-seq"])
    df['data'] = np.repeat(data_label, [rna_embeddings.shape[0], atac_embeddings.shape[0]], axis=0)
    df['predicted'] = label_dic[labels.astype(int)]
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "data",
        palette = sns.color_palette("tab10", 2),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig("output/p1.png")
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "predicted",
        palette = sns.color_palette("Set2", 7),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig("output/p2.png")