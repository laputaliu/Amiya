'''
 masked huber loss + corr-based loss
 mean + var loss
 add min-max norm after log and medium correction
 using huber loss and gradient clipping
 using softmax at the output
 no freeze
'''

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import random
import os
import pickle as pkl
import re
import datetime
from tqdm import tqdm
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GELU, Sigmoid, MSELoss, Parameter, HuberLoss, CosineSimilarity
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINConv, GATv2Conv
from torch_geometric.nn.norm import BatchNorm, GraphNorm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, mask_feature, mask_to_index, index_to_mask, to_dense_batch, select
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation, SetTransformerAggregation

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import ConcordanceCorrCoef
from lightning.pytorch.profilers import PyTorchProfiler

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)



class EnhancedBipartiteTransform(nn.Module):
    def __init__(self, n_rxn: int, n_met: int, S_init: torch.Tensor, edge_index: torch.Tensor,
                 num_heads: int=4, dropout: float=0.5, S_res_weight: float=1.0):
        """
        Args:
            n_rxn (int): Number of reactions.
            n_met (int): Number of metabolites.
            S_init (torch.Tensor): Initial transformation matrix [n_rxn, n_met].
            edge_index (torch.Tensor): Edge indices for the bipartite graph [2, num_edges].
            num_heads (int): Number of attention heads (default: 4).
            dropout (float): Dropout rate (default: 0.1).
            S_res_weight (float): Weight of S transformation residual connection [0,1], set to 0 -> no prior S modification
        """
        super(EnhancedBipartiteTransform, self).__init__()
        self.n_rxn = n_rxn
        self.n_met = n_met
        self.num_heads = num_heads
        self.S_res_weight = S_res_weight

        # Self-attention on reactions
        # Input dim is n_rxn (reaction features), output dim remains n_rxn
        self.self_attn = nn.MultiheadAttention(embed_dim=n_rxn, num_heads=num_heads, dropout=dropout)

        # Graph attention layer (GATConv)
        # Input channels = n_rxn (reaction features), output channels = n_met / num_heads (per head)
        self.gat = GATv2Conv(
            # in_channels=n_rxn,
            in_channels=1,
            # out_channels=n_met // num_heads,  # Ensure output size matches n_met after concatenation
            out_channels=1,
            heads=num_heads,
            dropout=dropout,
            concat=False,
            add_self_loops=False,
            negative_slope=0.2
        )

        # Transformation matrix S as a learnable parameter, initialized with prior knowledge
        self.S = nn.Parameter(S_init.clone())

        # Edge indices for the bipartite graph (reaction-to-metabolite connections)
        self.edge_index = edge_index

        self.gelu = GELU()
        self.sigmoid = Sigmoid()


    def forward(self, X_rxn: torch.Tensor):
        """
        Args:
            X_rxn (torch.Tensor): Reaction features [n_spot, n_rxn].
        
        Returns:
            X_met (torch.Tensor): Metabolite features [n_spot, n_met].
        """
        # Step 1: Self-attention on reactions
        # Reshape X_rxn to [1, n_spot, n_rxn] for MultiheadAttention
        X_rxn = X_rxn.unsqueeze(0)  # [1, n_spot, n_rxn]
        # print('> forward > X_rxn: {}'.format(X_rxn.size()))
        attn_output, attn_weight = self.self_attn(X_rxn, X_rxn, X_rxn)
        # print('> forward > attn_output: {}'.format(attn_output.size()))
        X_rxn_attn = self.gelu(attn_output.squeeze(0))  # [n_spot, n_rxn]
        # print('> forward > X_rxn_attn: {}'.format(X_rxn_attn.size()))

        # Step 2: Prepare for graph attention
        # Since GATConv operates on node features and edges, we need to adjust for n_spot
        # Flatten X_rxn_attn to treat each spot's reactions as separate nodes
        n_spot = X_rxn_attn.size(0)
        X_rxn_flat = X_rxn_attn.view(n_spot * self.n_rxn)  # [n_spot * n_rxn]
        # print('> forward > X_rxn_flat: {}'.format(X_rxn_flat.size()))

        # Create dummy metabolite features (zeros) for the bipartite graph
        dummy_met_features = torch.zeros(n_spot * self.n_met, device=X_rxn.device)  # [n_spot * n_met]
        # print('> forward > dummy_met_features: {}'.format(dummy_met_features.size()))
        X_bg = torch.cat([X_rxn_flat, dummy_met_features], dim=0).view(-1,1)  # [n_spot * (n_rxn + n_met)]
        # print('> forward > X_bg: {}'.format(X_bg.size()))

        # Adjust edge_index for multiple spots
        # Original edge_index is [2, num_edges] for one graph; replicate for n_spot
        edge_index_base = self.edge_index.to(X_rxn.device)  # [2, num_edges]
        # print('> forward > edge_index_base: {}'.format(edge_index_base.size()))
        # print('> forward > edge_index_base device: {}'.format(edge_index_base.device))
        offset = torch.arange(n_spot, device=edge_index_base.device) * (self.n_rxn + self.n_met)
        edge_index_offset = edge_index_base.unsqueeze(0) + offset.view(-1, 1, 1)  # [n_spot, 2, num_edges]
        edge_index_flat = edge_index_offset.view(2, -1)  # [2, n_spot * num_edges]
        # print('> forward > edge_index_flat: {}'.format(edge_index_flat.size()))

        # print('> forward > edge_index_base device: {}'.format(edge_index_base.device))
        # print('> forward > edge_index_flat device: {}'.format(edge_index_flat.device))
        # print('> forward > X_bg device: {}'.format(X_bg.device))
        # Step 3: Apply graph attention
        gat_output, gat_weight_pack = self.gat(X_bg, edge_index_flat, return_attention_weights=True)  # [n_spot * (n_rxn + n_met), n_met]
        # Extract metabolite features (last n_spot * n_met rows)
        X_met_flat = self.gelu(gat_output[n_spot * self.n_rxn:])  # [n_spot * n_met]
        X_met = X_met_flat.view(n_spot, self.n_met)  # [n_spot, n_met]

        # Step 4: Incorporate S transformation as a residual connection
        X_met_linear = torch.matmul(X_rxn_attn, self.S)  # [n_spot, n_met]
        X_met = X_met + self.S_res_weight * X_met_linear  # Combine GAT output with linear transformation

        return (self.sigmoid(X_met), X_rxn, X_rxn_attn, attn_weight, gat_weight_pack[1])

    def l1_regularization(self):
        """
        Returns:
            torch.Tensor: L1 norm of the S matrix for regularization.
        """
        return torch.sum(torch.abs(self.S))


class FineTuningConfig:
    n_nei = 6
    nlayer_nei = 2
    batch_size = 64
    accumu_step = 5
    num_workers = 20
    
    nhead_outer = 1
    dropout_outer = 0.5
    S_res_weight = 0
    freeze_layers = False
    small_lr_layers = None

    max_epochs = 500
    lr = 0.005
    weight_decay = 1e-5 # weight decay
    lr_small = 0.000001
    
    accelerator='gpu'
    devices = [1,2]
    # precision = "16-mixed"
    precision = "32-true"

    root_dir = './v0171_fine_gork_masked_corr_huber_loss_minmax_norm_r3/'
    # pre_train_ckpt = '/data/home/liuzhaoyang/ES/project/MetaSpace/large_ST/v0171/pre/v0171_pre_gork/Amiya_./v0171_pre_gork/version_3/checkpoints/best-epoch=13-Loss/val=0.54.ckpt'
    pre_train_ckpt = '/data/home/liuzhaoyang/ES/project/MetaSpace/large_ST/v0171/pre/v0171_pre_gork_r2/Amiya_./v0171_pre_gork_r2/version_0/checkpoints/best-epoch=5-Loss/val=0.54.ckpt'
    
    test_size = 0.2 
    val_size = 0.1
    drop_last = True

    ref_tag = 'maldi'
    reg_weight = 1e-5
    lw_corr_mean = 0.5
    lw_corr = 0.5
    lw_corr_col = 0.5
    
    record_flag = True


class PreTrainingConfig:

    n_nei = 6
    nlayer_nei = 2
    batch_size = 64
    accumu_step = 5
    num_workers = 10
    
    nfeature_embed = 256
    nlayer_ST_graph = 6
    nlayer_met_graph = 6
    nhead_SGAT = 4
    masking_rate = 0.7
    replace_rate = 0.2

    max_epochs = 100
    lr = 0.001
    weight_decay = 1e-5 # weight decay
    
    accelerator='gpu'
    devices = [1,7]
    precision = "16-mixed"

    root_dir = './v0171_pre_gork_lr_bs_optimal/'
    
    test_size = 0.2 
    val_size = 0.1
    drop_last = True



class LitAmiyaFineTune(L.LightningModule):
    def __init__(self, pre_trained_lit_amiya: L.LightningModule, met_edge_index: torch.Tensor,
                 n_rxn: int, n_met: int, S_init: torch.Tensor, rm_edge_index: torch.Tensor, index_mapping_list: list,
                 num_heads: int=4, dropout: float=0.2, S_res_weight: float=1.0, reg_weight: float=0.01, lw_corr_mean: float=0.3, lw_corr: float=0.5, lw_corr_col: float=0.5,
                 freeze_layers: bool=True, small_lr_layers: list=None, lr_small: float=0.0001, lr: float=0.0005, wd: float=1e-5,
                  batch_size: int=64, record_flag=True):
        super().__init__()
        self.lw_corr_mean = lw_corr_mean
        self.lw_corr = lw_corr
        self.lw_corr_col = lw_corr_col
        self.reg_weight = reg_weight
        self.index_mapping_list = index_mapping_list
        self.save_hyperparameters()
        self.register_buffer('met_edge_index', met_edge_index.t().contiguous())
        self.register_buffer('rm_edge_index', rm_edge_index)
        self.register_buffer('S_init', S_init)
        # self.register_buffer('index_mapping_list', index_mapping_list)

        # Extract the core Amiya model from the pre-trained LitAmiya
        self.amiya = pre_trained_lit_amiya.amiya

        self.huber_loss = HuberLoss(delta=5.0)

        # Freeze pre-trained layers (except the new output layer)
        if freeze_layers:
            for name, param in self.amiya.named_parameters():
                # Only allow updates to rxn2gene by excluding it from freezing
                if 'rxn2gene' not in name:
                    param.requires_grad = False

        # Replace the output layer (self.rxn2gene) with a new MLP
        # in_features = self.amiya.rxn2gene.in_features  # Get input size from the original layer
        self.amiya.rxn2gene = self._build_new_outer(n_rxn, n_met, S_init, self.rm_edge_index, num_heads, dropout, S_res_weight)

        # Store layers that should have a smaller learning rate (optional)
        self.small_lr_layers = small_lr_layers or []

    def _build_new_outer(self, n_rxn: int, n_met: int, S_init: torch.Tensor, rm_edge_index: torch.Tensor,
                         num_heads: int=4, dropout: float=0.2, S_res_weight: float=1.0):
        """
        Builds the enhanced MLP module with all requested features.
    
        Args:
            n_rxn (int): Number of reactions.
            n_met (int): Number of metabolites.
            S_init (torch.Tensor): Initial transformation matrix [n_rxn, n_met].
            edge_index (torch.Tensor): Edge indices for the bipartite graph [2, num_edges].
            num_heads (int): Number of attention heads (default: 4).
            dropout (float): Dropout rate (default: 0.1).
        
        Returns:
            EnhancedBipartiteTransform: The enhanced module.
        """
        return EnhancedBipartiteTransform(n_rxn, n_met, S_init, self.rm_edge_index, num_heads, dropout, S_res_weight)

    def _reorder_graph(self, graph: Data, sorted_idx: torch.Tensor) -> Data:
        # # print('> _reorder_graph > graph.batch: {}'.format(graph.batch.size()))
        # # print('> _reorder_graph > graph.x: {}'.format(graph.x.size()))
        # # print('> _reorder_graph > sorted_idx: {}'.format(sorted_idx.size()))
        idx_mapping = torch.zeros_like(sorted_idx, device=self.device)
        # # print('> _reorder_graph > idx_mapping: {}'.format(idx_mapping.size()))
        idx_mapping[sorted_idx] = torch.arange(len(sorted_idx), device=self.device)
        graph.x = graph.x[sorted_idx]
        graph.pos = graph.pos[sorted_idx]
        graph.slide = graph.slide[sorted_idx]
        graph.edge_index = idx_mapping[graph.edge_index]
        graph.batch = graph.batch[sorted_idx]

        if hasattr(graph, 'rxn_exp'):
            graph.rxn_exp = graph.rxn_exp[sorted_idx]
        if hasattr(graph, 'met_ref'):
            # ground-truth metabolite intensity used in fine-tuning
            graph.met_ref = graph.met_ref[sorted_idx]
        return graph

    def _prepare_met_edges(self) -> torch.Tensor:
        offsets = torch.arange(
            0, self.hparams.batch_size * self.hparams.n_rxn, self.hparams.n_rxn, device=self.device
        ).repeat_interleave(self.met_edge_index.size(1))
        return (self.met_edge_index.repeat(1, self.hparams.batch_size) + offsets).contiguous()

    
    def forward(self, graph: Data) -> tuple:
        # # print('> LitAmiya_forward > graph.batch: {}'.format(graph.batch.size()))
        sorted_idx = torch.argsort(graph.batch)
        graph = self._reorder_graph(graph, sorted_idx)
        met_batch = torch.repeat_interleave(
            torch.arange(self.hparams.batch_size, device=self.device), self.hparams.n_rxn
        )
        # # print('> LitAmiya_forward > met_batch: {}'.format(met_batch.size()))
        met_edge_index = self._prepare_met_edges()
        # # print('> LitAmiya_forward > met_edge_index: {}'.format(met_edge_index.size()))
        return self.amiya(
            x=graph.x.to(self.device), edge_index=graph.edge_index.to(self.device),
            rxn_exp=None,
            met_edge_index=met_edge_index, batch=graph.batch.to(self.device),
            batch_size=self.hparams.batch_size, met_batch=met_batch
        )


    def _shared_step(self, graph: Data, prefix: str) -> torch.Tensor:
        X_met_pack, mask_info = self(graph)
        X_met, X_rxn, X_rxn_attn, attn_weight, gat_weight = X_met_pack
        X_ref = graph.met_ref

        loss = self._compute_loss(X_met, X_ref, graph['graph_idx'], prefix)

        selected_X_met, selected_X_ref = self._select_share_met(X_met, X_ref, graph['graph_idx'])
        self._update_metrics(selected_X_met, selected_X_ref, prefix)
        if self.hparams.record_flag:
            self.log_dict({
                f'Loss/{prefix}': loss,
                'lr': self.trainer.optimizers[0].param_groups[0]['lr']
            }, sync_dist=True, on_step=True)
            if self.global_step % 50 == 0:
                self.logger.experiment.add_histogram("Input_X_rxn", X_rxn, self.global_step)
                self.logger.experiment.add_histogram("Input_selfattn_Layer1", X_rxn_attn, self.global_step)
                self.logger.experiment.add_histogram("Input_selfattn_weight_Layer1", attn_weight, self.global_step)
                self.logger.experiment.add_histogram("Output_gat_weight_Layer1", gat_weight, self.global_step)
                self.logger.experiment.add_histogram("Output_X_met", X_met, self.global_step)
        return loss


    def _select_share_met(self, X_met: torch.Tensor, X_ref: torch.Tensor, graph_idx: list):
        
        # Convert graph_idx to tensor if it’s a list
        if isinstance(graph_idx, list):
            graph_idx = torch.tensor(graph_idx, device=self.device)
    
        # Lists to store selected columns for all spots
        selected_X_met = []
        selected_X_ref = []
    
        # Process each unique graph in the batch
        unique_graphs = torch.unique(graph_idx)
        for k in unique_graphs:
            # Find spots belonging to graph k
            spot_mask = (graph_idx == k)
            spots_k = spot_mask.nonzero(as_tuple=True)[0]  # Indices of spots in graph k
    
            # Get the index mapping for graph k
            index_map_k = self.index_mapping_list[int(k)]
            if not index_map_k:
                continue  # Skip if no shared metabolites for this graph
    
            # Extract column indices for shared metabolites
            idx_met = list(index_map_k.keys())    # Columns in X_met
            idx_ref = list(index_map_k.values())  # Corresponding columns in X_ref
    
            # Select the relevant columns for this graph’s spots
            X_met_k = X_met[spots_k][:, idx_met]  # Shape: [num_spots_k, num_shared_met]
            X_ref_k = X_ref[spots_k][:, idx_ref]  # Shape: [num_spots_k, num_shared_met]
    
            # Store the selected tensors
            selected_X_met.append(X_met_k)
            selected_X_ref.append(X_ref_k)
    
        # Check if any shared metabolites were found
        if not selected_X_met:
            raise ValueError("No shared metabolites found in any graph.")
    
        # Concatenate all selected tensors across graphs
        selected_X_met = torch.cat(selected_X_met, dim=0)  # Shape: [total_selected_spots, num_shared_met]
        selected_X_ref = torch.cat(selected_X_ref, dim=0)  # Shape: [total_selected_spots, num_shared_met]

        return selected_X_met, selected_X_ref


    def _compute_corr(self, X_met: torch.Tensor, X_ref: torch.Tensor, wise: str='col', eps: float=1e-8):
        # met-wise
        n_met = X_met.size(1) if wise=='col' else X_met.size(0)
        correlations = []
    
        # Loop over each metabolite
        for m in range(n_met):
            # Extract prediction and reference vectors for metabolite m
            pred_m = X_met[:, m] if wise== 'col' else X_met[m, :]
            ref_m = X_ref[:, m] if wise== 'col' else X_ref[m, :]

            # skip zero var to avoid NaN
            if pred_m.std() > 0 and ref_m.std() > 0:
                # Compute correlation with epsilon
                corr = torch.corrcoef(torch.stack([pred_m, ref_m], dim=0))
                std_pred = pred_m.std(unbiased=False) + eps
                std_ref = ref_m.std(unbiased=False) + eps
                rho = corr[0, 1] / (std_pred * std_ref)
                rho = rho.clamp(-1, 1)  # Ensure within bounds
                correlations.append(rho)

        # Handle case where all metabolites are skipped
        if len(correlations) == 0:
            print('!!! All met with 0 var')
            return torch.tensor(0.0, device=X_met_pred.device)
    
        # Stack correlations into a tensor and compute the average loss
        correlations = torch.stack(correlations)  # Shape: [n_met]
        corr = 1 - correlations.mean()   # Scalar
        if torch.isnan(corr):
            print("NaN detected in corr!")

        loss_corr = corr

        # align_dim = 0 if wise=='col' else 1
        # # Mean alignment
        # loss_mean = torch.pow((X_met.mean(dim=align_dim) - X_ref.mean(dim=align_dim)), 2).mean()
        # # Variance alignment
        # loss_var = torch.pow((X_met.var(dim=align_dim) - X_ref.var(dim=align_dim)), 2).mean()

        # loss_corr = (1-self.lw_corr_mean)*corr + self.lw_corr_mean*(loss_mean + loss_var)

        # if torch.isnan(loss_mean):
        #     print("NaN detected in loss_mean!")
        # if torch.isnan(loss_var):
        #     print("NaN detected in loss_var!")
        # if torch.isnan(loss_corr):
        #     print("NaN detected in loss_corr!")

        return loss_corr

        
    
    def _compute_loss(self, X_met: torch.Tensor, X_ref: Tensor, graph_idx: torch.Tensor, prefix: str) -> torch.Tensor:
        """
        Compute the loss between X_met and X_ref based on shared metabolites for each graph in a batch.
    
        Args:
            X_met (torch.Tensor): Predicted tensor of shape [n_spot, n_met].
            X_ref (torch.Tensor): Reference tensor of shape [n_spot, n_met_ref].
            graph_idx (torch.Tensor or list): Indicates which graph each spot belongs to, length [n_spot].
            index_mapping_list (list): List of dictionaries, where index_mapping_list[k] = {idx_met: idx_ref, ...}
                                       maps X_met columns to X_ref columns for graph k.
    
        Returns:
            torch.Tensor: The computed loss (scalar).
        """
        # Ensure all tensors are on the same device
        device = X_met.device
        n_spot = X_met.size(0)
    
        # ---------------------------------- #
        # sce loss for met with ground-truth #
        # ---------------------------------- #
        selected_X_met, selected_X_ref = self._select_share_met(X_met, X_ref, graph_idx)

        if torch.isnan(selected_X_met).any() or torch.isnan(selected_X_ref).any():
            print("NaN detected in inputs!")
    
        # Compute the loss 
        # loss_ref = self.amiya.sce_loss(selected_X_met, selected_X_ref)
        loss_ref = self.huber_loss(selected_X_met, selected_X_ref)
        if torch.isnan(loss_ref):
            print("NaN detected in loss_ref!")

        # --------- #
        # corr loss #
        # --------- #
        # met-wise
        loss_corr_col = self._compute_corr(selected_X_met, selected_X_ref, wise='col')
        # spot-wise
        loss_corr_row = self._compute_corr(selected_X_met, selected_X_ref, wise='row')
        loss_corr = self.lw_corr_col*loss_corr_col + (1-self.lw_corr_col)*loss_corr_row

        # -------------- #
        # regularization #
        # -------------- #
        reg_loss = self.amiya.rxn2gene.l1_regularization()
        if torch.isnan(reg_loss):
            print("NaN detected in reg_loss!")

        self.log_dict({
            f'Loss_ref/{prefix}': (1-self.lw_corr)*loss_ref.item(),
            f'Loss_corr/{prefix}': self.lw_corr*loss_corr.item(),
            f'_loss_corr_row/{prefix}': loss_corr_row.item(),
            f'_loss_corr_col/{prefix}': loss_corr_col.item(),
            f'Loss_reg/{prefix}': self.reg_weight*reg_loss.item()
        }, prog_bar=False, sync_dist=True, on_step=True)

    
        return (1-self.lw_corr)*loss_ref + self.lw_corr*loss_corr + self.reg_weight*reg_loss


    def _update_metrics(self, pred: torch.Tensor, target: torch.Tensor, prefix: str):
        # # print('> _update_metrics > pred: {}'.format(pred.size()))
        # # print('> _update_metrics > target: {}'.format(target.size()))
        metrics_spot = ConcordanceCorrCoef(num_outputs=pred.size(0)).to(pred.device)
        metrics_met = ConcordanceCorrCoef(num_outputs=pred.size(1)).to(pred.device)

        cos_spot = CosineSimilarity(dim=1).to(pred.device)
        cos_met = CosineSimilarity(dim=0).to(pred.device)

        # metrics_spot.update(pred.T, target.T)
        # metrics_met.update(pred, target)
        
        self.log_dict({
            f'Corr_spot/{prefix}': torch.nanmedian(metrics_spot(pred.T, target.T)),
            f'Corr_feature/{prefix}': torch.nanmedian(metrics_met(pred, target)),
            f'CosSim_spot/{prefix}': torch.nanmedian(cos_spot(pred.T, target.T)),
            f'CosSim_feature/{prefix}': torch.nanmedian(cos_met(pred, target))
        }, prog_bar=True, sync_dist=True, on_step=True)

        metrics_spot.reset()
        metrics_met.reset()
        # cos_spot.reset()
        # cos_met.reset()


    
    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'test')

    
    def configure_optimizers(self):
        """Configure optimizer with optional differential learning rates."""
        if self.small_lr_layers:
            # Use different learning rates for specified layers
            param_groups = [
                # Small LR for layers allowing slight changes
                {'params': [p for n, p in self.amiya.named_parameters() 
                           if n in self.small_lr_layers], 'lr': self.hparams.lr_small, 'weight_decay': self.hparams.wd},
                # Normal LR for the new output layer (and any other trainable params)
                {'params': [p for n, p in self.amiya.named_parameters() 
                           if n not in self.small_lr_layers and p.requires_grad], 'lr': self.hparams.lr, 'weight_decay': self.hparams.wd}
            ]
        else:
            # Single LR for all trainable parameters (e.g., just the new MLP)
            param_groups = [p for p in self.amiya.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(param_groups, lr=self.hparams.lr, weight_decay=self.hparams.wd) 
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10, T_mult=2, eta_min=1e-7
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', "frequency": 4}
        }


def met_id_mapping(query_ids, query_key, target_key,
                  mapID_ref='/data/home/liuzhaoyang/ES/project/MetaSpace/met_model/Amiya_met_meta.tsv'):
    met_meta_df = pd.read_csv(mapID_ref, sep='\t', index_col=0)
    if query_key not in met_meta_df.columns or target_key not in met_meta_df.columns:
        raise ValueError("query or target key not found in avi met meta resources")

    single_id = isinstance(query_ids, str)
    if single_id:
        query_ids = [query_ids]  # Convert single ID to a list for uniform processing
    elif not isinstance(query_ids, list):
        raise ValueError("query_ids must be a string or a list of strings")

    sub_df = met_meta_df[met_meta_df[query_key].isin(query_ids)]

    # Group by source_resource and collect unique, non-NaN target IDs as lists
    result = sub_df.groupby(query_key)[target_key].apply(
        lambda x: x.unique().item(0)
    ).to_dict()

    # Return format depends on whether a single ID or multiple IDs were provided
    if single_id:
        # For a single ID, return the IDs (NAN if not found)
        return result.get(query_ids[0], 'NA')
    else:
        # For multiple IDs, return a dictionary with all source_ids, mapping to their target IDs
        return {sid: result.get(sid, 'NA') for sid in query_ids}



def find_common_items(S1, S2):
    """
    Find common items between two pandas Series, S1 and S2, and return their first matching indices.
    Handles NaN and 'NA' values by excluding them from the matching process.

    Parameters:
    - S1: pandas Series
    - S2: pandas Series

    Returns:
    - dict: A dictionary where keys are indices from S1 and values are the first matching indices from S2
    """
    # Create copies of S1 and S2 to avoid modifying the originals
    S1_copy = S1.copy()
    S2_copy = S2.copy()

    # Replace 'NA' strings with NaN for uniform handling
    S1_copy = S1_copy.replace('NA', np.nan)
    S2_copy = S2_copy.replace('NA', np.nan)

    # Drop NaN values to focus on valid items
    S1_valid = S1_copy.dropna()
    S2_valid = S2_copy.dropna()

    # Find unique items in S1_valid to avoid redundant checks
    unique_items_S1 = S1_valid.unique()

    # Initialize a dictionary to store the first matching indices
    index_dict = {}

    # Iterate over unique items in S1_valid
    for item in unique_items_S1:
        # Check if the item exists in S2_valid
        if item in S2_valid.values:
            # Get the first index in S1 where the item appears
            S1_index = S1_valid[S1_valid == item].index[0]
            # Get the first index in S2 where the item appears
            S2_index = S2_valid[S2_valid == item].index[0]
            # Store the index pair in the dictionary
            index_dict[S1_index] = S2_index

    return index_dict


def preprocess_maldi_data(X_met_ref, force_normalize=False):
    """
    Preprocess MALDI reference data with log transformation and spot-wise median normalization.
    Auto-detects if the data has already been normalized based on negative values and median centering.

    Args:
        X_met_ref (torch.Tensor): Reference MALDI data of shape [n_spot, n_met]
        force_normalize (bool): If True, always apply normalization regardless of auto-detection

    Returns:
        torch.Tensor: Preprocessed data X_met_ref_normalized
    """
    # Auto-detect if data is already normalized
    has_negative = (X_met_ref < 0).any()
    row_medians = X_met_ref.median(dim=1).values
    is_median_centered = torch.allclose(row_medians, torch.zeros_like(row_medians), atol=1e-5)

    # If data has negative values or is median-centered, assume it's already normalized
    if not force_normalize and (has_negative or is_median_centered):
        print("Data appears to be already normalized. Skipping preprocessing.")
        return X_met_ref

    # Step 1: Log transformation (using log1p to handle zero values)
    X_met_ref_log = torch.log1p(X_met_ref)

    # Step 2: Spot-wise median normalization
    spot_medians = X_met_ref_log.median(dim=1, keepdim=True).values
    X_met_ref_normalized = X_met_ref_log - spot_medians

    # Step 3: min-max scale
    X_min = X_met_ref_normalized.min(dim=1, keepdim=True).values  # Shape: [n_spot, 1]
    X_max = X_met_ref_normalized.max(dim=1, keepdim=True).values  # Shape: [n_spot, 1]
    # Compute the range for each spot
    range_X = X_max - X_min
    # Avoid division by zero: set range to 1 where min == max
    range_X[range_X == 0] = 1.0
    # Scale each spot independently
    X_met_ref_normalized_scaled = (X_met_ref_normalized - X_min) / range_X
    # For spots with constant values, set to 0
    X_met_ref_normalized_scaled[range_X.squeeze() == 0] = 0.0

    return X_met_ref_normalized_scaled


def data_preparing(graph_list: str, met_edge: str, S_file: str, avi_genes: str,
                   graph_metID_key: str='met_HMDB_ID', query_metID_key: str=None, target_metID_key: str='KEGG',
                  mapID_ref: str='/data/home/liuzhaoyang/ES/project/MetaSpace/met_model/Amiya_met_meta.tsv', ref_tag: str='maldi',
                  verbose: bool=True):
    """
    additional step in fine-tuning: go through all graphs to decide the avi met in met_ref
    """
    
    graph_list = torch.load(graph_list)
    
    # ------------------- #
    # S & metabolic graph #
    # ------------------- #
    with open(met_edge, 'rb') as f:
        met_edge = torch.tensor(pkl.load(f))
    
    # loading metabolic model info.
    S = pd.read_csv(S_file, sep='\t', index_col=0)
    S_tensor = torch.from_numpy(S.to_numpy()).float().T
    S_keggID = met_id_mapping(list(S.index), 'recon_id_no_comp','KEGG')
    
    # ------------------------- #
    # S metabolite HMDB mapping #
    # ------------------------- #
    mapped_id_dict_list, index_mapping_list = [], []
    if query_metID_key:
        for tmp_i, tmp_g in enumerate(graph_list):
            query_ids = tmp_g[graph_metID_key]
            mapped_id_dict = met_id_mapping(query_ids, query_metID_key, target_metID_key)
            common_index_dict = find_common_items(
                pd.Series(S_keggID.values()),
                pd.Series(mapped_id_dict.values())
            )
            mapped_id_dict_list.append(mapped_id_dict)
            index_mapping_list.append(common_index_dict)

            graph_list[tmp_i]['graph_idx'] = torch.Tensor([tmp_i]).repeat(tmp_g['x'].size(0))
            
            if verbose:
                print('graph {}/{}: {} met mapped, {} met found in avi met model'.format(tmp_i+1, len(graph_list)+1, len(mapped_id_dict.keys()), len(common_index_dict.keys())))
    else:
        mapped_id_dict = {}
        for tmp_i, tmp_g in enumerate(graph_list):
            common_index_dict = find_common_items(
                pd.Series(S_keggID.values()),
                pd.Series(tmp_g[graph_metID_key])
            )
            mapped_id_dict_list.append(mapped_id_dict)
            index_mapping_list.append(common_index_dict)

            graph_list[tmp_i]['graph_idx'] = torch.Tensor([tmp_i]).repeat(tmp_g['x'].size(0))

            if verbose:
                print('graph {}/{}: {} met mapped, {} met found in avi met model'.format(tmp_i+1, len(graph_list)+1, tmp_g[graph_metID_key].size(0), len(common_index_dict.keys())))
    
    with open(avi_genes, 'r') as f:
        avi_genes = f.read().strip().split('\n')

    # generate rxn-met Bipartite graph edge index
    rm_edge_index = generate_edge_index_from_S(S_tensor, S_tensor.size(0))

    # ------------------------------ #
    # normalize reference MALDI data #
    # ------------------------------ #
    for tmp_i in range(len(graph_list)):
        graph_list[tmp_i][ref_tag] = preprocess_maldi_data(graph_list[tmp_i][ref_tag])

    return graph_list, met_edge, S, S_tensor, avi_genes, mapped_id_dict_list, index_mapping_list, rm_edge_index


def setup_training(cfg: FineTuningConfig, pre_trained_lit_amiya: L.LightningModule, graph_list: list, met_edge: torch.Tensor,
                   n_rxn: int, n_met: int, S_init: torch.Tensor, rm_edge_index: torch.Tensor, index_mapping_list: list):

    train_val_graphs, test_graphs = train_test_split(
        graph_list, 
        test_size=cfg.test_size,
        random_state=42
    )
    train_graphs, val_graphs = train_test_split(
        train_val_graphs,
        test_size=cfg.val_size/(1-cfg.test_size),
        random_state=42
    )

    train_data = merge_graphs(train_graphs, cfg.ref_tag)
    val_data = merge_graphs(val_graphs, cfg.ref_tag)
    test_data = merge_graphs(test_graphs, cfg.ref_tag)

    # init fine-tuning model
    model = LitAmiyaFineTune(
        pre_trained_lit_amiya=pre_trained_lit_amiya,
        met_edge_index=met_edge,
        n_rxn=n_rxn,
        n_met=n_met,
        S_init=S_init,
        rm_edge_index=rm_edge_index,
        index_mapping_list=index_mapping_list,
        num_heads=cfg.nhead_outer,
        dropout=cfg.dropout_outer,
        S_res_weight=cfg.S_res_weight,
        reg_weight=cfg.reg_weight,
        lw_corr_mean=cfg.lw_corr_mean,
        lw_corr=cfg.lw_corr,
        lw_corr_col=cfg.lw_corr_col,
        freeze_layers=cfg.freeze_layers,
        small_lr_layers=cfg.small_lr_layers,
        lr_small=cfg.lr_small,
        lr=cfg.lr,
        wd=cfg.weight_decay,
        batch_size=cfg.batch_size,
        record_flag=cfg.record_flag
    )
        
    return model, train_data, val_data, test_data


def get_callbacks():
    return [
        # 模型检查点
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="Loss/val",
            mode="min",
            # save_top_k=2,
            every_n_epochs=1,
            filename="best-{epoch}"
        ),
        
        # 早停机制
        L.pytorch.callbacks.EarlyStopping(
            monitor="Loss/val",
            patience=25,
            mode="min"
        ),
        
        # 学习率监控
        L.pytorch.callbacks.LearningRateMonitor()
    ]


def merge_graphs(graph_list: list, ref_tag: str='maldi') -> Data:
    """
    将多个图合并为单个大图
    :param graph_list: 图数据列表[Data]
    :return: 合并后的Data对象
    """
    offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes for g in graph_list[:-1]]),
        dim=0
    )
    
    # 合并节点特征
    x = torch.cat([g.x for g in graph_list], dim=0)
    
    # 合并边索引
    edge_indices = []
    for i, g in enumerate(graph_list):
        edge_indices.append(g.edge_index + offsets[i])
    edge_index = torch.cat(edge_indices, dim=1)
    
    # 创建批次标识
    batch = torch.cat([
        torch.full((g.num_nodes,), i) 
        for i, g in enumerate(graph_list)
    ])

    pos = torch.cat([g.pos for g in graph_list], dim=0)

    if hasattr(graph_list[0], 'graph_idx'):
        graph_idx = torch.cat([g.graph_idx for g in graph_list], dim=0)
    else:
        graph_idx = torch.Tensor([0]).repeat(len(graph_list))
    
    # if hasattr(graph_list[0], 'rxn_exp'):
    #     rxn_exp = torch.cat([g.rxn_exp for g in graph_list], dim=0)
    #     return Data(x=x, edge_index=edge_index, slide=batch, pos=pos, rxn_exp=rxn_exp, graph_idx=graph_idx)
    if hasattr(graph_list[0], ref_tag):
        # ground-truth metabolite intensity used in fine-tuning
        met_ref = torch.cat([g[ref_tag] for g in graph_list], dim=0)
        return Data(x=x, edge_index=edge_index, slide=batch, pos=pos, met_ref=met_ref, graph_idx=graph_idx)

    
    return Data(x=x, edge_index=edge_index, pos=pos, slide=batch, graph_idx=graph_idx)


def generate_edge_index_from_S(S_init, n_rxn):
    """
    Generate edge_index for the bipartite graph based on non-zero entries in S_init.

    Args:
        S_init (torch.Tensor): Tensor of shape [n_rxn, n_met], where non-zero entries indicate edges.
        n_rxn (int): Number of reactions.

    Returns:
        torch.Tensor: edge_index of shape [2, num_edges], with edges from reactions to metabolites.
                      - First row: reaction indices (0 to n_rxn-1).
                      - Second row: metabolite indices (n_rxn to n_rxn + n_met - 1).
    """
    # Find indices of non-zero elements in S_init
    nonzero_indices = torch.nonzero(S_init, as_tuple=False)  # Shape: [num_edges, 2]

    # Construct edge_index
    edge_index = torch.stack([
        nonzero_indices[:, 0],           # Reaction indices (i)
        nonzero_indices[:, 1] + n_rxn    # Metabolite indices (j + n_rxn)
    ], dim=0)  # Shape: [2, num_edges]

    return edge_index



def main():
    
    cfg = FineTuningConfig()
    pre_cfg = PreTrainingConfig()
    

    # ------------------------------------ #
    # change this in final release version #
    # ------------------------------------ #
    import sys
    sys.path.append('/data/home/liuzhaoyang/ES/project/MetaSpace/large_ST/v0171/pre/')
    from v0171_pre_base_gork3_refine_r2 import LitAmiya, MyDataLoader
    # ------------------------------------ #
    # change this in final release version #
    # ------------------------------------ #

    graph_list, met_edge, S, S_tensor, avi_genes, mapped_id_dict_list, index_mapping_list, rm_edge_index = data_preparing(
        graph_list = '/data/home/liuzhaoyang/data/GBM_Spatial_MultiOmics/GBM_sct_counts_graph_list.pth',
        met_edge = '/data/home/liuzhaoyang/ES/project/MetaSpace/edge_index_without_com_dedup.pkl',
        S_file = '/data/home/liuzhaoyang/ES/project/MetaSpace/S_without_com_trim_dedup.tsv',
        avi_genes = '/data/home/liuzhaoyang/ES/project/MetaSpace/avi_genes.txt',
        graph_metID_key = 'met_HMDB_ID',
        query_metID_key='HMDB',
        target_metID_key='KEGG',
        ref_tag=cfg.ref_tag,
        verbose=True
    )
    n_gene = len(avi_genes)
    n_rxn = S_tensor.size(0)
    n_met = S_tensor.size(1)
    logger = TensorBoardLogger(save_dir=cfg.root_dir, name="Amiya_{}".format(cfg.root_dir))


    # print(graph_list[0])
    
    # reloading pre-trained model
    pre_trained_lit_amiya = LitAmiya.load_from_checkpoint(
        cfg.pre_train_ckpt,
        met_edge_index=met_edge,
        lr=pre_cfg.lr,
        wd=pre_cfg.weight_decay,
        record_flag=True,
        n_gene=n_gene,
        n_rxn=n_rxn,
        nfeature_embed=pre_cfg.nfeature_embed,
        nlayer_ST_graph=pre_cfg.nlayer_ST_graph,
        nlayer_met_graph=pre_cfg.nlayer_met_graph,
        nhead_SGAT=pre_cfg.nhead_SGAT,
        batch_size=pre_cfg.batch_size,
        # accumu_step=pre_cfg.accumu_step,
        masking_rate=pre_cfg.masking_rate,
        replace_rate=pre_cfg.replace_rate
    )

    model, train_data, val_data, test_data = setup_training(cfg, pre_trained_lit_amiya, graph_list, met_edge, n_rxn, n_met, S_tensor, rm_edge_index, index_mapping_list)

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        callbacks=get_callbacks(),
        accumulate_grad_batches=4,
        strategy="ddp_find_unused_parameters_true",  # ddp_find_unused_parameters_true
        # strategy='fsdp',
        precision=cfg.precision,
        log_every_n_steps=5,
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=cfg.root_dir,
        gradient_clip_val=1.0
    )

    
    trainer.fit(
        model,
        train_dataloaders=MyDataLoader(
            train_data,
            num_neighbors=[cfg.n_nei] * cfg.nlayer_nei, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last
        ),
        val_dataloaders=MyDataLoader(
            val_data,
            num_neighbors=[cfg.n_nei] * cfg.nlayer_nei,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last
        )
    )

    trainer.test(
        model,
        dataloaders=MyDataLoader(
            test_data,
            num_neighbors=[cfg.n_nei] * cfg.nlayer_nei,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last
        )
    )

    torch.save(model.amiya.state_dict(), "{}/final_amiya_model.pth".format(cfg.root_dir))



if __name__ == '__main__':

    torch.set_float32_matmul_precision('high')
    L.seed_everything(42)

    with torch.cuda.amp.autocast():
        main()

