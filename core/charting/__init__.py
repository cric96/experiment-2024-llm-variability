import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def create_chart(embeddings, reducer, where, axis=None, alpha=0.5, title=None, xlim=None, ylim=None, size=5,
                 labels=None):
    result = reducer.fit_transform(embeddings)
    plt.figure(figsize=(size, size)) if not axis else None
    plot_chart(axis if axis else plt, result, alpha, title, xlim, ylim, labels)
    plt.savefig(where) if where else {}
    plt.show() if not axis else None


def plot_chart(plot, result, alpha, title, xlim, ylim, labels=None):
    if plot is plt:
        sns.scatterplot(x=result[:, 0], y=result[:, 1], hue=labels, legend='full', alpha=alpha)
        plt.title(title)
    else:
        sns.scatterplot(x=result[:, 0], y=result[:, 1], hue=labels, legend='full', alpha=alpha, ax=plot)
        plot.set_title(title)
    plot.legend() if labels else None
    if xlim: plot.set_xlim(-xlim, xlim)
    if ylim: plot.set_ylim(-ylim, ylim)


def pca_chart(embeddings, where, axis=None, alpha=0.5, title=None, xlim=None, ylim=None, size=5, labels=None):
    create_chart(embeddings, PCA(n_components=2), where, axis, alpha, title, xlim, ylim, size, labels)


def pca_compare(embeddings_a, embeddings_b, color='gray', alpha_a=0.1, alpha_b=0.5, title=None, xlim=None, ylim=None,
                size=5, labels=None):
    pca = PCA(n_components=2)
    result_A = pca.fit_transform(embeddings_a)
    result_B = pca.fit_transform(embeddings_b)
    plt.figure(figsize=(size, size))
    plt.scatter(result_A[:, 0], result_A[:, 1], c=color, alpha=alpha_a)
    plt.scatter(result_B[:, 0], result_B[:, 1], c='blue', alpha=alpha_b)
    plt.title(title) if title else None
    if xlim: plt.xlim(-xlim, xlim)
    if ylim: plt.ylim(-ylim, ylim)
    plt.show()


def tsne_chart(embeddings, where, axis=None, alpha=0.5, title=None, xlim=None, ylim=None, size=5, labels=None):
    create_chart(embeddings, TSNE(n_components=2, random_state=42), where, axis, alpha, title, xlim, ylim, size, labels)


def umap_chart(embeddings, where, n_neighbors=15, min_dist=0.1, metric='euclidean', axis=None, alpha=0.5, title=None,
               xlim=None, ylim=None, size=5, labels=None):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    create_chart(embeddings, reducer, where, axis, alpha, title, xlim, ylim, size, labels)
