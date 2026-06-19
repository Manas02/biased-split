"""Implementation of Molecular Network"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity


mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_ecfp4_bitvect(smi): 
    return mfpgen.GetFingerprint(Chem.MolFromSmiles(smi))

def smiles_to_ecfp4_np(smi): 
    return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))

def compute_similarity_matrix(fps_bitvect):
    n = len(fps_bitvect)
    sim_matrix = np.eye(n, dtype=np.float32)

    for i in range(n - 1): # Remember symmetrical matrix !
        sims = BulkTanimotoSimilarity(fps_bitvect[i], fps_bitvect[i+1:])
        sim_matrix[i, i+1:] = sims
        sim_matrix[i+1:, i] = sims
        
    return sim_matrix

def df_to_ecfp4_molecular_network(df, smiles_col, activity_col, similarity_threshold):
    fps_bitvect = df[smiles_col].map(smiles_to_ecfp4_bitvect).tolist()
    sim_matrix = compute_similarity_matrix(fps_bitvect)

    adj_matrix = np.triu(sim_matrix, k=1)
    adj_matrix[adj_matrix < similarity_threshold] = 0
    G = nx.from_numpy_array(adj_matrix)

    node_attrs = {
        n: {'smiles': smi, 'activity': act} 
        for n, (smi, act) in enumerate(zip(df[smiles_col], df[activity_col]))
    }
    nx.set_node_attributes(G, node_attrs)

    return G





