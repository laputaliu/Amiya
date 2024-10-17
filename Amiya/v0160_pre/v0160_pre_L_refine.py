import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import random
import os
import pickle as pkl
import re
import datetime
from tqdm import tqdm


import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, TransformerEncoder, TransformerEncoderLayer, CosineEmbeddingLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from torcheval.metrics import MulticlassAUROC
from sklearn.model_selection import KFold

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINConv, GATConv, SuperGATConv, aggr
from torch_geometric.nn.norm import BatchNorm, GraphNorm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import ConcordanceCorrCoef

import importlib
import sys
sys.path.append('/mnt/Venus/home//liuzhaoyang/project/MetaSpace/')
import processing as MSpp
import model as MS

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:82944"



#########
# model #
#########

class Amiya(torch.nn.Module):
    def __init__(
        self, 
        n_gene,
        n_rxn,
        nfeature_embed = 512,
        nlayer_ST_graph = 2,
        nlayer_met_graph = 2,
        nlayer_concat_encoder = 3,
        nfeature_met=13,
        nhead_SGAT=4,
        nhead_mapping=4,
        GIN_eps = 0,
        MLP_dr = 0.1,
        MLP_act = 'gelu',
        MLP_norm = 'GraphNorm',
        train_GIN_eps = False
    ):
        '''
        n_rxn: num of reactions in metabolic model
        nfeature_embed: the dim of features of ST & metabolic embedded tensors. (the tensors used for concat)
        nfeature_met: the feature number of input metabolic tensor (13 neighbors: 4 nei 2 lops)

        
        Gene-reaction mappint part:
            just a simple two layer NN for predicting reaction 'exp' from gene exp with g-r relation mask;
            will try more complex model later
            - gr_nchan_in: gene-reaction mapping input dim: num. of input gene
        GraphSAGE part:
            
        '''
        super().__init__()

        self.n_gene = n_gene
        self.n_rxn = n_rxn
        self.nfeature_embed = nfeature_embed
        self.nfeature_met = nfeature_met
        self.nhead_SGAT = nhead_SGAT
        self.nhead_mapping = nhead_mapping

        self.input_GN = GraphNorm(self.n_gene)
        
        self.identity = Linear(in_features=self.n_gene, out_features=self.nfeature_embed)
        self.identity_met = Linear(in_features=self.n_gene, out_features=self.n_rxn)

        # self.gene_rxn_map = GATConv(
        #     in_channels=(1,1),
        #     out_channels=1,
        #     heads=self.nhead_mapping,
        #     concat=False,
        #     dropout=0.1,
        #     add_self_loops=True
        # )

        self.gene_rxn_map = Linear(
            in_features=self.n_gene,
            out_features=self.n_rxn
        )

        self.ST_GIN_layers = torch.nn.ModuleList()
        for tmp_l in range(nlayer_ST_graph):
            if tmp_l==0:
                self.ST_GIN_layers.append(
                    GINConv(
                        nn=MLP(
                            [self.n_rxn, 1024, 512, 128, self.nfeature_embed],
                            dropout=MLP_dr,
                            act=MLP_act,
                            norm=MLP_norm
                        ),
                        eps=GIN_eps,
                        train_eps=train_GIN_eps
                    )
                )
            else:
                self.ST_GIN_layers.append(
                    GINConv(
                        nn=MLP(
                            [self.nfeature_embed, 256, 128, 256, self.nfeature_embed],
                            dropout=MLP_dr,
                            act=MLP_act,
                            norm=MLP_norm
                        ),
                        eps=GIN_eps,
                        train_eps=train_GIN_eps
                    )
                )
                

        # ------------------- #
        # try SuperGAT layers #
        # ------------------- #
        self.met_SGAT_layers = torch.nn.ModuleList()
        for tmp_l in range(nlayer_met_graph):
            if tmp_l < (nlayer_met_graph-1):
                self.met_SGAT_layers.append(
                    SuperGATConv(
                        in_channels=self.nfeature_met,
                        out_channels=self.nfeature_met,
                        heads=self.nhead_SGAT,
                        concat=False,
                        dropout=0.1,
                        attention_type='MX',
                        edge_sample_ratio=1,
                        is_undirected=False,
                        add_self_loops=False
                    )
                )
            else:
                self.met_SGAT_layers.append(
                    SuperGATConv(
                        # the in channels here used to be nfeature_embed
                        in_channels=self.nfeature_met,
                        out_channels=self.nfeature_met, # just output 1 dim
                        heads=self.nhead_SGAT,
                        concat=False,
                        dropout=0.1,
                        attention_type='MX',
                        edge_sample_ratio=1,
                        is_undirected=False,
                        add_self_loops=False
                    )
                )


        # readout function in met garph
        self.met_SGAT_aggrs = torch.nn.ModuleList()
        for tmp_l in range(nlayer_met_graph):
            self.met_SGAT_aggrs.append(aggr.SoftmaxAggregation(learn=True))

        # reshape feature number of met embedding to self.nfeature_embed
        self.met_compress = MLP([self.n_rxn, 512, 256, self.nfeature_embed],
                                dropout=MLP_dr,
                                act=MLP_act,
                                norm=MLP_norm
                               )

        
        # -- mapping st & metabolite concat to output 
        self.embed_rxn_map = MLP(
            # in_channels=self.nfeature_embed*(nlayer_ST_graph+nlayer_met_graph),
            [self.nfeature_embed*2, 128, 512, 1024, self.n_rxn],
            dropout=0.1,
            act='gelu',
            norm='GraphNorm'
        )

        # self.rxn_gene_map = GATConv(
        #     in_channels=(1,1),
        #     out_channels=1,
        #     heads=self.nhead_mapping,
        #     concat=False,
        #     dropout=0.1,
        #     add_self_loops=True
        # )
        
        self.rxn_gene_map = Linear(
            in_features=self.n_rxn,
            out_features=self.n_gene
        )
               
    def forward(self, x, edge_index, rxn_exp, gr_edge_index, met_edge_index, batch, batch_size, met_batch):
        '''
        x: (spot, rxn)
        edge_index: (2, st_edge)
        met_edge_index: (2, 63936)
        batch: (spot) e.g. [0,0,1,2,2,...]
        batch_size: the actual batch size in each lop
        '''

        # input normalize
        x = self.input_GN(x, batch=batch)
        # print('x shape {}'.format(x.shape))
        x_identity = F.gelu(self.identity(x))
        x_identity_met = F.gelu(self.identity_met(x))
        # print('x_identity shape {}'.format(x_identity.shape))4


        # ------------------- #
        # mapping gene to rxn #
        # ------------------- #
        # init_size = x.size()
        # x = x.reshape(x.size(0), x.size(1), 1)
        # rxn_exp = rxn_exp.reshape(rxn_exp.size(0), rxn_exp.size(1), 1)
        
        # x_rxn, gr_atten = self.gene_rxn_map(
        #     x=(x, rxn_exp),
        #     edge_index=gr_edge_index,
        #     # edge_weight=gr_graph.edge_weight,
        #     return_attention_weights=True
        # )
        # x_rxn = x_rxn.reshape(init_size[0], self.n_rxn)

        x_rxn = F.gelu(self.gene_rxn_map(x))

        
        ##########################
        # embedding spatial info #
        ##########################
        for tmp_l, gin in enumerate(self.ST_GIN_layers):
            st_embed = F.gelu(gin(torch.add(st_embed, x_identity), edge_index)) if tmp_l > 0 else F.gelu(gin(x_rxn, edge_index))
            # print('st_embed shape {}'.format(st_embed.shape))
                        
            # concat each layer's graph readout
            if tmp_l == 0:
                st_concat = F.gelu(global_add_pool(torch.add(st_embed, x_identity), batch))
                # (128, 512)
            else:
                st_concat = torch.add(
                        st_concat, 
                        F.gelu(global_add_pool(st_embed, batch))
                )
                # (128, 512)
            # print('st_concat shape {}'.format(st_concat.shape))
            

        # print('>>>\n')
        # print(batch)
        
        #############################
        # embedding metabolite info #
        #############################
        # padding & reshape 
        x_chunk = torch.split(x_rxn, tuple(torch.unique(batch, return_counts = True)[1]))
        x_chunk = tuple(map(lambda x: torch.transpose(F.pad(x_rxn, (0, 0, 0, (self.nfeature_met - x_rxn.size(0)))), 0, 1), x_chunk))
        x_met = torch.cat(x_chunk, dim=0)
        # print('x_met shape {}'.format(x_met.shape))

        x_chunk_met = torch.split(x_identity_met, tuple(torch.unique(batch, return_counts = True)[1]))
        x_chunk_met = tuple(map(lambda x: torch.transpose(F.pad(x_rxn, (0, 0, 0, (self.nfeature_met - x_rxn.size(0)))), 0, 1), x_chunk_met))
        x_identity_met = torch.cat(x_chunk_met, dim=0)        


        # ---------------- #
        # SuperGAT version #
        # ---------------- #
        for tmp_l, sgat in enumerate(self.met_SGAT_layers):
            # print('x_met: ', x_met.get_device())
            # print('met_edge_index: ', met_edge_index.get_device())
            # print('met_batch: ', met_batch.get_device())
            # print('x_identity_met: ', x_identity_met.get_device())

            # print(x_met.size(), met_edge_index.size(), met_batch.size())
            
            met_embed = F.gelu(sgat(x_met, met_edge_index, batch=met_batch)) if tmp_l == 0 else F.gelu(sgat(torch.add(met_embed, x_identity_met), met_edge_index, batch=met_batch))
            # met_embed = torch.reshape(met_embed, (batch_size, self.n_rxn, self.nfeature_met))
            # print('met_embed shape {}'.format(met_embed.shape))

            # return met_embed
                        
            # concat each layer's graph readout
            if tmp_l == 0:
                met_concat = F.gelu(self.met_SGAT_aggrs[tmp_l](torch.add(met_embed, x_identity_met), dim=-1).reshape((batch_size, self.n_rxn)))
            else:
                met_concat = torch.add(met_concat, F.gelu(self.met_SGAT_aggrs[tmp_l](met_embed, dim=-1).reshape((batch_size, self.n_rxn))))
        met_concat = F.gelu(self.met_compress(met_concat)) # make the nFeature of met_concat same as the ST_concat
            # if tmp_l == 0:
            #     met_concat = global_add_pool(torch.add(met_embed, x_identity).relu(), batch)
            # else:
            #     met_concat = torch.add(
            #             met_concat, 
            #             global_add_pool(torch.add(met_embed, x_identity).relu(), batch)
            #     )
            # print('met_concat shape {}'.format(met_concat.shape))

        # return met_concat

        ######### !!!!!!!!!!! why use batch_size as the dim of st_concat & met_concat !!!!!!!! ############
        
        ##########################
        # concat ST & meta embed #
        ##########################
        # st_out = self.tf_encoder(torch.cat((st_concat,met_concat), dim=1))  # so why we used a tf encoder here? ....
        # st_out = self.linear_map_rxn(st_out).relu()
        
        out_rxn = F.gelu(self.embed_rxn_map(torch.cat((st_concat,met_concat), dim=1)))


        # ------------------- #
        # mapping rxn to gene #
        # ------------------- #
        # out_rxn = out_rxn.reshape(out_rxn.size(0), out_rxn.size(1), 1)
        # out_gene, rg_atten = self.rxn_gene_map(
        #     x=(out_rxn, x),
        #     edge_index=gr_edge_index.flip([0]),
        #     # edge_weight=gr_graph.edge_weight,
        #     return_attention_weights=True
        # )
        # out_gene = out_gene.reshape(init_size[0], self.n_gene)

        # return out_gene, gr_atten, rg_atten

        out_gene = self.rxn_gene_map(out_rxn).relu()
        
        return out_gene



class MyDataLoader(NeighborLoader):
    def __init__(self,
                 graph_list,
                 num_neighbors,
                 batch_size,
                 disjoint=True,
                 num_workers=1,
                 pin_flag=True,
                 **kwargs
                ):
        self.graph_list = graph_list
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.disjoint = disjoint
        self.num_workers = num_workers
        self.pin_flag = pin_flag
        super().__init__(
            data=self.graph_list,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            disjoint=self.disjoint,
            num_workers=self.num_workers,
            pin_memory=self.pin_flag
            # **kwargs
        )



class LitAmiya(L.LightningModule):
    def __init__(self,
                 met_edge_index,
                 lr,
                 wd,
                 record_flag,
                 n_gene,
                 n_rxn,
                 nfeature_embed = 256,
                 nlayer_ST_graph=2,
                 nlayer_met_graph=2,
                 nhead_SGAT=4,
                 nhead_mapping=4,
                 nlayer_concat_encoder=4,
                 n_nei = 6,
                 nlayer_nei = 2,
                 batch_size = 64,
                 disjoint = True,
                 pin_flag=True,
                 model_checkpoint='',
                 freeze=False,
                 n_fold=0, # KFold iter
                 accumu_step=5
                ):
        super().__init__()
        # self.met_edge_index = torch.tensor(met_edge_index, dtype=torch.long, device=self.device).t().contiguous()
        self.met_edge_index = met_edge_index
        self.lr = lr
        self.wd = wd
        self.record = record_flag
        self.n_gene = n_gene
        self.n_rxn = n_rxn
        self.nfeature_embed = nfeature_embed
        self.nlayer_ST_graph = nlayer_ST_graph
        self.nlayer_met_graph = nlayer_met_graph
        self.nhead_SGAT = nhead_SGAT
        self.nhead_mapping = nhead_mapping
        self.nlayer_concat_encoder = nlayer_concat_encoder
        self.n_fold = n_fold
        self.n_nei = n_nei
        self.nlayer_nei = nlayer_nei
        self.num_neighbors = [n_nei] * nlayer_nei
        self.batch_size = batch_size
        self.disjoint = disjoint
        self.pin_flag = pin_flag
        self.accumu_step = accumu_step

        self.loss_recon = MSELoss(reduction='mean')

        self.amiya = Amiya(
            self.n_gene,
            self.n_rxn,
            nfeature_embed=self.nfeature_embed,
            nlayer_ST_graph = self.nlayer_ST_graph,
            nlayer_met_graph = self.nlayer_met_graph,
            nlayer_concat_encoder = self.nlayer_concat_encoder,
            nhead_SGAT=self.nhead_SGAT,
            nhead_mapping=self.nhead_mapping
        )
        if len(model_checkpoint) > 0:
            self.amiya.load_state_dict(torch.load(model_checkpoint))

        if freeze:
            for param in self.amiya.parameters():
                param.requires_grad = False



    def forward(self, tmp_graph):

        # print('>>>\n')
        # print(tmp_graph)

        met_batch = torch.arange(0,tmp_graph.batch_size, device=self.device).repeat_interleave(self.n_rxn)
        met_edge_index = torch.tensor(self.met_edge_index, dtype=torch.long, device=self.device).t().contiguous()

        # print('met_edge_index_forward_L: ', met_edge_index.get_device())
        
        st_out = self.amiya(
            x=tmp_graph.x,
            edge_index=tmp_graph.edge_index,
            rxn_exp=tmp_graph.rxn_exp,
            gr_edge_index=tmp_graph.gr_edge_index,
            met_edge_index=met_edge_index,
            batch=tmp_graph.batch,
            batch_size=tmp_graph.batch_size,
            met_batch=met_batch,
        )

        # return st_out, gr_atten, rg_atten
        return st_out

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = LinearLR(
            optimizer,
            start_factor=1/100,
            end_factor=1,
            # total_iters=int((n_epoch_PT*200)*0.8), # n_iter in 200 epochs version is ~ 14000
            total_iters = 10000
            # total_steps=stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    
    def train_dataloader(self, graph_list):
        loader = MyDataLoader(
            graph_list=graph_list,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            disjoint=self.disjoint,
            pin_flag=self.pin_flag
        )
        return loader


    def validation_dataloader(self, graph_list):
        loader = MyDataLoader(
            graph_list=graph_list,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            disjoint=self.disjoint,
            pin_flag=self.pin_flag
        )
        return loader

    
    def training_step(self, tmp_graph, n_batch,
                      accumu_step=5
                     ):

        # out_embed, gr_atten, rg_atten = self.forward(tmp_graph)
        out_embed = self.forward(tmp_graph)
        raw_embed = tmp_graph.x[:tmp_graph.batch_size]
        loss = self.loss_recon(out_embed, raw_embed) + self.loss_recon(out_embed.t(), raw_embed.t())

        if self.record:
            self.log('loss_recon/train/Fold-{}'.format(self.n_fold), loss,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # adding SGAT loss
        for tmp_gi in range(self.nlayer_met_graph):
            loss += self.amiya.met_SGAT_layers[tmp_gi].get_attention_loss()
            if self.record:
                self.log('loss_SGAT_{}/train/Fold-{}'.format(tmp_gi, self.n_fold),
                         self.amiya.met_SGAT_layers[tmp_gi].get_attention_loss(),
                         on_epoch=True, sync_dist=True, batch_size=self.batch_size)                

        # spot-wise corr
        corr_cal = ConcordanceCorrCoef(num_outputs=out_embed.size(0)).to(self.device)
        spot_corr = torch.nanmedian(corr_cal(out_embed.t().detach(), raw_embed.t()))
        # metabolite-wise corr
        corr_cal = ConcordanceCorrCoef(num_outputs=out_embed.size(1)).to(self.device)
        met_corr = torch.nanmedian(corr_cal(out_embed.detach(), raw_embed))
    
        if self.record:
            self.log('Loss/train/Fold-{}'.format(self.n_fold), loss,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
            self.log('Corr_spot/train/Fold-{}'.format(self.n_fold), spot_corr,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
            self.log('Corr_met/train/Fold-{}'.format(self.n_fold), met_corr,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        
        return loss


    def validation_step(self, tmp_graph, n_batch):
        
        # out_embed, gr_atten, rg_atten = self.forward(tmp_graph)
        out_embed = self.forward(tmp_graph)
        raw_embed = tmp_graph.x[:tmp_graph.batch_size]
        loss = self.loss_recon(out_embed, raw_embed)

        # no SGAT loss in val step
        
        # spot-wise corr
        corr_cal = ConcordanceCorrCoef(num_outputs=out_embed.size(0)).to(self.device)
        spot_corr = torch.nanmedian(corr_cal(out_embed.t().detach(), raw_embed.t()))
        # metabolite-wise corr
        corr_cal = ConcordanceCorrCoef(num_outputs=out_embed.size(1)).to(self.device)
        met_corr = torch.nanmedian(corr_cal(out_embed.detach(), raw_embed))
        
        if self.record:
            self.log('Loss/val/Fold-{}'.format(self.n_fold), loss,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
            self.log('Corr_spot/val/Fold-{}'.format(self.n_fold), spot_corr,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
            self.log('Corr_met/val/Fold-{}'.format(self.n_fold), met_corr,
                     on_epoch=True, sync_dist=True, batch_size=self.batch_size, prog_bar=True)



# def build_gene_rxn_bip(rxn_meta_dic):
#     rxn_list = list(rxn_meta_dic.keys())
#     gene_list, gene_index, rxn_index = [], [], []
#     for tmp_ri, tmp_r in enumerate(rxn_list):
#         tmp_genes = rxn_meta_dic[tmp_r]['gene_association']
#         if len(tmp_genes) == 0:
#             continue
#         else:
#             for tmp_g in tmp_genes:
#                 if tmp_g in gene_list:
#                     rxn_index.append(tmp_ri)
#                     gene_index.append(gene_list.index(tmp_g))
#                 else:
#                     rxn_index.append(tmp_ri)
#                     gene_index.append(len(gene_list))
#                     gene_list.append(tmp_g)

#     return gene_index, rxn_index


if __name__ == '__main__':
        
    
    ##################
    # hyperparameter #
    ##################
    train_val_frac = 0.8
    k_folds = 5
    n_nei = 6
    nlayer_nei = 2
    batch_size = 128
    accumu_step = 5
    nfeature_embed = 256
    
    nlayer_ST_graph = 2
    nlayer_met_graph = 2
    nlayer_concat_encoder = 4  # N layers of transfromer encoders attached after concated embeddings
    
    # pre-train
    n_epoch_PT = 100
    lr_PT = 0.0005
    wd_PT = 0.0000001 # weight decay
    # fine-tune
    n_epoch_FT = 200
    lr_FT = 0.0001
    wd_FT = 0.00001
    
    model_save_step = 1
    
    pct_start = 0.3
    div_factor = 100
    final_div_factor = 1000
    
    torch.set_float32_matmul_precision('high')

    save_model_state = True
    record = True
    run_comment = "{}_v0160_pretrain_L".format(datetime.datetime.now().strftime('%y%m%d-%H%M'))

    random_seed = 42
    L.pytorch.seed_everything(random_seed, workers=True)
        
    
    ################
    # data loading #
    ################
    
    
    # ------------------- #
    # S & metabolic graph #
    # ------------------- #
    
    with open('/mnt/Venus/home//liuzhaoyang/project/MetaSpace/metabolic_model/RECON2_2/edge_index_without_com_dedup.pkl', 'rb') as f:
        met_edge_index = pkl.load(f)
    # met_edge_index = torch.tensor(met_edge_index, dtype=torch.long).t()
    # met_edge_index = torch.tensor(met_edge_index, dtype=torch.long).t().contiguous()
    
    # loading metabolic model info.
    S = pd.read_csv('/mnt/Venus/home//liuzhaoyang/project/MetaSpace/metabolic_model/RECON2_2/S_without_com_trim_dedup.tsv', sep='\t', index_col=0)
    S_tensor = torch.from_numpy(S.to_numpy())
    S_tensor = S_tensor.float()
    S_tensor = S_tensor
    
    
    # ------------------------- #
    # S metabolite HMDB mapping #
    # ------------------------- #
    
    with open('/mnt/Venus/home//liuzhaoyang/data/GBM_Spatial_MultiOmics/annotated_metabolite_HMDB_ID.txt', 'r') as f:
        avi_meta = f.read().strip().split('\n')
    
    S_meta_mapping = MSpp.build_S_meta_mapping(S, avi_meta, with_compartment=False, verbose=True,
                                              met_md_path='/mnt/Venus/home/liuzhaoyang/project/MetaSpace/metabolic_model/RECON2_2/met_md.csv')
    
    
    # ------------------------- #
    # train & test data loading #
    # ------------------------- #
    
    graph_list = torch.load('/data/home/liuzhaoyang/data/EcoTIME/EcoTIME_human_tumor_raw_graph_list.pth')
    
    
    # ------------------ #
    # gene rxn bipartite #
    # ------------------ #
    with open('/data/home/liuzhaoyang/ES/project/MetaSpace/rxn_meta_dic.pkl', 'rb') as f:
        rxn_meta_dic = pkl.load(f)
        
    with open('/data/home/liuzhaoyang/ES/project/MetaSpace/avi_genes.txt', 'r') as f:
        avi_genes = f.read().strip().split('\n')
    n_gene = len(avi_genes)
        
    rxn_list = list(S.columns)

    # # build gene rxn bipartite
    # gene_index, rxn_index = [], []
    # for tmp_ri, tmp_r in enumerate(rxn_list):
    #     tmp_genes = rxn_meta_dic[tmp_r]['gene_association']
    #     if len(tmp_genes) == 0:
    #         continue
    #     else:
    #         for tmp_g in tmp_genes:
    #             if tmp_g in avi_genes:
    #                 rxn_index.append(tmp_ri)
    #                 gene_index.append(avi_genes.index(tmp_g))
    #             else:
    #                 continue
    # gr_graph = Data(edge_index = torch.tensor([gene_index,rxn_index], dtype=torch.long),
    #                edge_weight = torch.ones(len(gene_index)))


    
    
    #########
    # KFold #
    #########

    train_test_sampler = DataLoader(graph_list, batch_size=int(np.ceil(train_val_frac*len(graph_list))),
                                        shuffle=False)
    train_val_dataset, test_dataset = train_test_sampler

    kfold_iter = KFold(n_splits=k_folds, shuffle=False)
    
    for n_fold, (train_ids, val_ids) in enumerate(kfold_iter.split(train_val_dataset)):
        
        # ---------- #
        # model init #
        # ---------- #
    
        # torch.manual_seed(random_seed)
        model = LitAmiya(
            met_edge_index,
            lr_PT,
            wd_PT,
            record,
            n_gene,
            S_tensor.shape[1],
            nlayer_ST_graph=nlayer_ST_graph,
            nlayer_met_graph=nlayer_met_graph,
            nlayer_concat_encoder=nlayer_concat_encoder,
            n_nei = n_nei,
            nlayer_nei = nlayer_nei,
            batch_size = batch_size,
            freeze=False,
            n_fold=n_fold, # KFold iter
            accumu_step=accumu_step
        )

        
        # --------------- #
        # dataloader init #
        # --------------- #            
        
        ##### using custom dataloader
        train_loader = model.train_dataloader(
            Batch.from_data_list(train_val_dataset[train_ids]),
            # train_val_dataset
            # num_neighbors=[n_nei] * nlayer_nei,
            # batch_size=batch_size
        )
        val_loader = model.validation_dataloader(
            # test_dataset
            Batch.from_data_list(train_val_dataset[val_ids]),
            # num_neighbors=[n_nei] * nlayer_nei,
            # batch_size=batch_size
        )


        # ---------------------- #
        # lightning trainer init #
        # ---------------------- #
        checkpoint_callback = ModelCheckpoint(
                filename="Amiya_0160_pre-L_{epoch:02d}",
                every_n_epochs=model_save_step,
                save_top_k=-1,
            )
    
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        logger = TensorBoardLogger(save_dir='./v0160_pre/', version=3, name="Amiya_0160_pre-L")
    
        trainer = L.Trainer(
            accelerator='gpu',
            devices=[6,7],
            strategy="fsdp",  # fsdp, ddp_find_unused_parameters_true
            max_epochs=n_epoch_PT,
            callbacks=[checkpoint_callback, lr_monitor],
            default_root_dir='./v0160_pre/',
            check_val_every_n_epoch=model_save_step,
            # deterministic=True,
            accumulate_grad_batches=accumu_step,
            logger=logger,
            profiler='simple'
        )    
                    
        trainer.fit(model, train_loader, val_loader)

    print('# --- finished --- #')
    
