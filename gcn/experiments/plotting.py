#%%
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from gcn.utils import load_citation_data


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_citation_data('cora')

G = nx.from_scipy_sparse_matrix(adj)

# largest connected component
Gc = max(nx.connected_component_subgraphs(G), key=len)
# relabel
mapping = dict(zip(Gc.nodes, range(len(Gc.nodes))))
Gc = nx.relabel_nodes(Gc, mapping)
pos = nx.spring_layout(Gc)


def eigvec(M, kth):
    val, vec = np.linalg.eigh(M)
    order = np.argsort(val)
    val = np.array(val[order])
    nonzero = np.where(val > 1e-6)
    lowest = np.min(nonzero)
    vec = np.array(vec[:, order])
    if kth < 0:
        return vec[:, kth]
    else:
        return vec[:, kth + lowest]


# eigenvector centrality
A = nx.adjacency_matrix(Gc).todense()
x = eigvec(A, -1)
x -= np.min(x)
x /= np.sum(x)
x = {i: x[i] for i in range(len(x))}

# eigenvectors of laplacian
L = nx.normalized_laplacian_matrix(Gc).todense()
y = eigvec(L, 0)
y = {i: y[i] for i in range(len(y))}


def draw(G, pos, measures, measure_name):
    cols = list(measures.values())
    labs = list(measures.keys())
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=cols, nodelist=labs,
                                   cmap=plt.cm.plasma, node_size=3)
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    edges = nx.draw_networkx_edges(G, pos=pos, alpha=0.6, edge_color='g')
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()


draw(Gc, pos, x, '1st eigenvector Laplacian')



