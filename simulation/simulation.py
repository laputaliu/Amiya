import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import random
import os
import pickle as pkl
import re
import datetime
from tqdm import tqdm
from typing import Optional
import argparse

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, TransformerEncoder, TransformerEncoderLayer, CosineEmbeddingLoss, MSELoss, Parameter
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from torcheval.metrics import MulticlassAUROC
from sklearn.model_selection import KFold

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINConv, GATConv, SuperGATConv, aggr, knn_graph
from torch_geometric.nn.norm import BatchNorm, GraphNorm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, mask_feature, mask_to_index, select, to_dense_batch, k_hop_subgraph
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import ConcordanceCorrCoef
from lightning.pytorch.profilers import PyTorchProfiler


import importlib
import sys
sys.path.append('/mnt/Venus/home//liuzhaoyang/project/MetaSpace/')
import processing as MSpp
import model as MS

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)



def slide_patching(
    tmp_graph,
    k=6 # the neibor number of each spot
    ):
    
    '''
    randomly subset ST slides, patching by n nei_layer with 6 nearest nei spots
    return subset spots & edge_index in subset
    '''
    
    # -------- #
    # patching #
    # -------- #
    
    # recompute the knn egde index
    knn_edge_index = knn_graph(
        tmp_graph['pos'],
        k=6, 
        loop=False
    )
    
    # randomly select target spot, ignore the boundary excluding
    target_index = int(torch.randperm(tmp_graph['x'].size(0))[0])
    subset, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
        target_index,
        patching_layer,
        knn_edge_index,
        relabel_nodes=True
    )

    return subset, subset_edge_index, mapping, edge_mask


def select_disturb_center(
    subset, # the output index tensor of slide_patching
    subset_edge_index,
    num_disturb_center=5,
    num_disturb_layer=3 # the number of influenced spot layers of a single disturbing center
):
    '''
    randomly select num_disturb_center spots as the disturbing centers in patched region
    return a dict record disturbed spots with their hop
    '''
    
    # randomly select over-expression central points, ignoring the boundary limitation
    disturb_centers = torch.randperm(subset.size(0))[:num_disturb_center]
    disturb_spots = {}
    for tmp_l in range(num_disturb_layer):
        if tmp_l == 0:
            tmp_spots = disturb_centers
            disturb_spots.update({
                tmp_l: disturb_centers
            })
        else:
            khop_subset, _, _, _ = k_hop_subgraph(
                disturb_centers,
                tmp_l,
                subset_edge_index
            )
            khop_spots = khop_subset[~np.isin(khop_subset, tmp_spots)]
            tmp_spots = khop_subset
            disturb_spots.update({
                tmp_l: khop_spots
            })

    return disturb_spots


def overexpress(
    tmp_graph,
    subset,
    subset_edge_index,
    disturb_spots, # the disturbed spot dict, the output of select_disturb_center
    met_gene_dic, # the met & gene lookup dict
    avi_genes, # all supported genes in current Amiya model
    num_disturb_met=None,
    num_disturb_layer=3
):
    '''
    randomly select met & corresponding genes to over-express
    return a graph
    '''
    # --------------- #
    # over-expression #
    # --------------- #
    # select some non-zero & high intensity metabolites to over-express
    # filtering avi met ()
    subset_maldi = tmp_graph['maldi'][subset]
    met_std, met_mean = torch.std_mean(subset_maldi, 0)
    avi_met = met_mean > 0 # non-zero
    # remove met with no gene associations
    avi_associa = [tmp_id for tmp_id in met_gene_dic.keys() if (len(met_gene_dic[tmp_id]['up']) + len(met_gene_dic[tmp_id]['down'])) > 0]
    avi_associa = np.isin(tmp_graph['met_HMDB_ID'], avi_associa)
    avi_met = avi_met & avi_associa
    # --- additional filtering can add here --- #
    avi_met = mask_to_index(avi_met)
    
    if num_disturb_met == None:
        num_disturb_met = avi_met.size(0)
        over_mets = avi_met
    else:
        num_disturb_met = min(avi_met.size(0), num_disturb_met)
        over_mets = avi_met[torch.randperm(avi_met.size(0))[:num_disturb_met]]
    
    # select corresponding genes to over-express
    over_met_ids = select(tmp_graph['met_HMDB_ID'], avi_met, dim=0)
    over_up_genes, over_down_genes = [], []
    for tmp_id in over_met_ids:
        over_up_genes += met_gene_dic[tmp_id]['up']
        over_down_genes += met_gene_dic[tmp_id]['down']
    over_up_genes = list(set(over_up_genes))
    over_down_genes = [tmp_g for tmp_g in set(over_down_genes) if tmp_g not in over_up_genes] # upstream genes first
    over_up_genes = mask_to_index(torch.tensor(np.isin(avi_genes, [tmp_g.upper() for tmp_g in over_up_genes])))
    over_down_genes = mask_to_index(torch.tensor(np.isin(avi_genes, [tmp_g.upper() for tmp_g in over_down_genes])))
    
    subset_x = tmp_graph['x'][subset]
    gene_std, gene_mean = torch.std_mean(subset_x, 0)
    
    # --- increase exp --- #
    # 0-hop: 3 mean; 1-hop: 2 mean ...
    # from outer to inner 
    for tmp_hop in reversed(range(num_disturb_layer)):
        tmp_spots = disturb_spots[tmp_hop]
        # over-expression met
        subset_maldi[tmp_spots,:][:,over_mets] += (num_disturb_layer-tmp_hop) * met_mean[over_mets]
        
        # over-expression gene
        # upstream
        subset_x[tmp_spots,:][:,over_up_genes] += (num_disturb_layer-tmp_hop) * gene_mean[over_up_genes]
        # downstream
        subset_x[tmp_spots,:][:,over_down_genes] -= (num_disturb_layer-tmp_hop) * gene_mean[over_down_genes]
        subset_x[tmp_spots,:][:,over_down_genes] = F.relu(subset_x[tmp_spots,:][:,over_down_genes])
    
    subset_graph = Data(
        x=subset_x,
        edge_index=subset_edge_index,
        pos=tmp_graph['pos'][subset],
        maldi=subset_maldi,
        met_HMDB_ID=tmp_graph['met_HMDB_ID'],
        over_mets=over_mets,
        over_up_genes=over_up_genes,
        over_down_genes=over_down_genes
    )

    return subset_graph





if __name__ == '__main__':

    # --- param & resources --- #
    parser = argparse.ArgumentParser(
        prog='simulation graph list generation', 
        description='simulate graphs for semi-simulated evaluation of CHIMERA, over-express specific metabolites & corresponding genes'
    )
    
    parser.add_argument('-n', '--n_round', default=100, type=int,
                        help='the number of simulation rounds')
    parser.add_argument('-c', '--n_center', default=5, type=int,
                        help='the number of distrubing centers')
    parser.add_argument('-l', '--n_layer', default=3, type=int,
                        help='the number of disturbing layers of each disturbing center')
    parser.add_argument('-m', '--n_met', default=20, type=int,
                        help='the number of over-expressed metabolites in each simulation turn')
    parser.add_argument('-d', '--outdir', default='./', type=str,
                        help='the output dir')
    parser.add_argument('-p', '--patch_layer', default=5, type=int,
                        help='the number of layers used in slide patching')

    # param below should not be changed
    parser.add_argument('--input_graphs', default='/data/home/liuzhaoyang/data/GBM_Spatial_MultiOmics/GBM_sct_counts_TIC_norm_graph_list.pth',
                        help='input graph list as simulating resources')
    parser.add_argument('--met_gene_dic', default='/data/home/liuzhaoyang/ES/project/MetaSpace/met_gene_dic.pkl',
                        help='met gene lookup dict path')
    parser.add_argument('--avi_genes', default='/data/home/liuzhaoyang/ES/project/MetaSpace/avi_genes.txt',
                        help='avi genes in CHIMERA metabolic model')


    args = parser.parse_args()

    simulation_round = args.n_round
    
    patching_layer = args.patch_layer
    num_disturb_center = args.n_center
    num_disturb_layer = args.n_layer # including center
    num_disturb_met = args.n_met

    save_flag = True
    save_dir = args.outdir
    
    graph_list = torch.load(args.input_graphs)
    
    # met & gene associations
    with open(args.met_gene_dic, 'rb') as f:
        met_gene_dic = pkl.load(f)
    
    # gene expression matrix name list 
    with open(args.avi_genes, 'r') as f:
        avi_genes = f.read().strip().split('\n')

    
    # --- simulation --- #
    simulate_graph_list = []
    for tmp_round in tqdm(range(simulation_round)):
        tmp_graph = graph_list[random.sample(range(len(graph_list)), 1)[0]]
        
        # --- pathching --- #
        subset, subset_edge_index, mapping, edge_mask = slide_patching(tmp_graph)
        
        # --- select disturbing centers --- #
        disturb_spots = select_disturb_center(
            subset,
            subset_edge_index,
            num_disturb_center=num_disturb_center,
            num_disturb_layer=num_disturb_layer
        )
    
        # --- over-exrpession --- #
        disturb_graph = overexpress(
            tmp_graph,
            subset,
            subset_edge_index,
            disturb_spots,
            met_gene_dic,
            avi_genes,
            num_disturb_met=num_disturb_met,
            num_disturb_layer=num_disturb_layer
        )

        simulate_graph_list.append(disturb_graph)

    
    # --- saving --- #
    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_tag = 'simul_center_{}_layer_{}_met_{}_round_{}'.format(
            num_disturb_center,
            num_disturb_layer,
            num_disturb_met,
            simulation_round
        )
        
        torch.save(
            simulate_graph_list,
            '{}/{}.pth'.format(save_dir, save_tag)
        )

        with open('{}/{}_spots.pkl'.format(save_dir, save_tag), 'wb') as f:
            pkl.dump(disturb_spots, f)
