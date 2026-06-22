"""Implementation of Molecular Network"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity, BulkTverskySimilarity

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_ecfp4_bitvect(smi):
    return mfpgen.GetFingerprint(Chem.MolFromSmiles(smi))


def smiles_to_ecfp4_np(smi):
    return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))


def compute_similarity_matrix(fps_bitvect, method="tanimoto"):
    alpha = 1
    beta = 0

    n = len(fps_bitvect)
    sim_matrix = np.eye(n, dtype=np.float32)

    for i in range(n - 1):
        target_fp = fps_bitvect[i]
        query_fps = fps_bitvect[i + 1 :]

        if method == "tanimoto":
            sims = BulkTanimotoSimilarity(target_fp, query_fps)
            sim_matrix[i, i + 1 :] = sims
            sim_matrix[i + 1 :, i] = sims

        elif method == "tversky":
            # Compute Tv(A, B) using standard alpha, beta
            sims_ab = BulkTverskySimilarity(target_fp, query_fps, alpha, beta)

            # Compute Tv(B, A) by swapping alpha and beta
            sims_ba = BulkTverskySimilarity(target_fp, query_fps, beta, alpha)

            # Get element-wise maximum for the two directions
            max_sims = np.maximum(sims_ab, sims_ba)

            # Assign symmetrically
            sim_matrix[i, i + 1 :] = max_sims
            sim_matrix[i + 1 :, i] = max_sims

    return sim_matrix


def molecular_network_from_list(
    smiless,
    activities,
    similarity_threshold,
    activity_threshold,
    similarity_method="tanimoto",
):
    fps_bitvect = [smiles_to_ecfp4_bitvect(smiles) for smiles in smiless]
    sim_matrix = compute_similarity_matrix(fps_bitvect, method=similarity_method)

    adj_matrix = np.triu(sim_matrix, k=1)
    adj_matrix[adj_matrix < similarity_threshold] = 0
    G = nx.from_numpy_array(adj_matrix)

    node_attrs = {
        n: {"smiles": smi, "activity": act}
        for n, (smi, act) in enumerate(zip(smiless, activities))
    }
    nx.set_node_attributes(G, node_attrs)
    G.graph["activity_label"] = "activity"
    G.graph["activity_threshold"] = activity_threshold
    G.graph["similarity_threshold"] = similarity_threshold
    G.graph["similarity_fp"] = "2048bit ECFP4"
    G.graph["similarity_distance"] = similarity_method
    return G


def df_to_ecfp4_molecular_network(
    df, smiles_col, activity_col, similarity_threshold, activity_threshold
):
    fps_bitvect = df[smiles_col].map(smiles_to_ecfp4_bitvect).tolist()
    sim_matrix = compute_similarity_matrix(fps_bitvect)

    adj_matrix = np.triu(sim_matrix, k=1)
    adj_matrix[adj_matrix < similarity_threshold] = 0
    G = nx.from_numpy_array(adj_matrix)

    node_attrs = {
        n: {"smiles": smi, "activity": act}
        for n, (smi, act) in enumerate(zip(df[smiles_col], df[activity_col]))
    }
    nx.set_node_attributes(G, node_attrs)
    G.graph["activity_label"] = activity_col
    G.graph["activity_threshold"] = activity_threshold
    G.graph["similarity_threshold"] = similarity_threshold
    G.graph["similarity_fp"] = "2048bit ECFP4"
    G.graph["similarity_distance"] = "Tanimoto"
    return G


def visualise_molnet(G, filepath=None):
    fig, ax = plt.subplots(figsize=(12, 9))

    pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")

    edge_colors = []
    for u, v in G.edges():
        if (
            abs(G.nodes[u]["activity"] - G.nodes[v]["activity"])
            > G.graph["activity_threshold"]
        ):
            edge_colors.append((1, 0, 0, 1))
        else:
            w = G.edges[u, v]["weight"]
            edge_colors.append((1 - w, 1 - w, 1 - w, 0.6))

    node_colors = [G.nodes[n]["activity"] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.8, ax=ax)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.Greys,
        node_size=40,
        linewidths=0,
        ax=ax,
    )

    cbar = fig.colorbar(nodes, ax=ax)
    cbar.set_label(G.graph["activity_label"])
    ax.axis("off")
    plt.title(
        f"Molecular Network with {G.number_of_nodes()} molecules & {G.number_of_edges()} edges made using \nSimilarity Threshold of {G.graph['similarity_threshold']} over {G.graph['similarity_fp']} fingerprints using {G.graph['similarity_distance']} Similarity"
    )
    if filepath:
        plt.tight_layout()
        plt.savefig(filepath)
    plt.show()


def visualise_molnet_split(
    G, train_idx, test_idx, effective_bias, intended_bias, filepath=None, cliff=True
):
    CLIFF = (0.65, 0.15, 0.20)
    TEST = (0.20, 0.40, 0.60)

    if "_pos" not in G.graph:
        G.graph["_pos"] = nx.nx_agraph.graphviz_layout(
            G, prog="sfdp", args="-Goverlap=false -GK=1.5"
        )
    pos = G.graph["_pos"]

    fig, ax = plt.subplots(figsize=(10, 9))
    edge_colors = []
    for u, v in G.edges():
        if (
            abs(G.nodes[u]["activity"] - G.nodes[v]["activity"])
            > G.graph["activity_threshold"]
        ):
            edge_colors.append(CLIFF + (0.9,))
        else:
            w = G.edges[u, v]["weight"]
            edge_colors.append((1 - w, 1 - w, 1 - w, 0.4))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.5, ax=ax)

    nodes_array = np.array(list(G.nodes()))
    train_nodes = nodes_array[train_idx]
    test_nodes = nodes_array[test_idx]
    activities = np.array([G.nodes[n]["activity"] for n in G.nodes()])
    vmin, vmax = activities.min(), activities.max()
    cmap = LinearSegmentedColormap.from_list(
        "greys_trunc", plt.cm.Greys(np.linspace(0.35, 0.95, 256))
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=train_nodes,
        node_color=[G.nodes[n]["activity"] for n in train_nodes],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        node_size=22,
        linewidths=0,
        ax=ax,
    )
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=test_nodes,
        node_color=[G.nodes[n]["activity"] for n in test_nodes],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        node_size=22,
        linewidths=0.9,
        edgecolors=TEST,
        ax=ax,
    )

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.018, pad=0.02, aspect=40)
    cbar.set_label(G.graph["activity_label"], fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.5",
            markersize=6,
            label="train",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.5",
            markeredgecolor=TEST,
            markeredgewidth=0.9,
            markersize=6,
            label="test",
        ),
    ]
    if cliff:
        handles.append(
            Line2D([0], [0], color=CLIFF, linewidth=1.2, label="activity cliff"),
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=3,
        fontsize=9,
        handletextpad=1.0,
        columnspacing=1.0,
    )

    caption = (
        f"{G.number_of_nodes()} molecules, {G.number_of_edges()} edges. "
        f"{G.graph['similarity_fp']}, {G.graph['similarity_distance']} ≥ {G.graph['similarity_threshold']}. "
        f"intended bias {intended_bias:.2f}, effective bias {effective_bias:.2f}"
    )
    fig.text(0.5, 0.045, caption, ha="center", fontsize=8, color="0.4")
    ax.axis("off")

    if filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
