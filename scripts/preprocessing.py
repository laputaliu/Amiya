
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import logging
import libsbml
import re
import pickle as pkl

import torch
from torch_geometric.data import Data
from pandarallel import pandarallel


# --------------------- #
# preprocessing ST data #
# --------------------- #


def prepare_ST(st_count_path, st_coord_path, 
               load_sep='\t', var_names='gene_symbols', transpose=True,
                sample_prefix=None,
               trans_to_rxn=True, rxn_meta_dic=None,
               nei_dis_ratio=1, n_nei=6,
               graph_x_source='sct',
               out_sct=True, out_raw=True, out_h5ad=True, save_dir='./'):
    '''
    load & preprocess downloaded ST dataset
    input: ST count file & coord path, or just a pandas Dataframe with row as gene, col as cell
    output: SCT count file, spatial graph pth
    
    - load_sep: used in loading txt format input
    - var_names: only used in load mtx
    - transpose: whether to transpose input matrix
    - sample_prefix: used to distinguish cell barcodes from diff. sample, will add before barcodes. if == None, then no prefix added
    - if trans_to_rxn==True & rxn_meta_dic!=None -> transfer gene expression into reaction expression according to the gene-rxn associations storaged in the rxn_meta_dic
    
    - neigh_dis_ratio: control distance for defining neighbor spots, (neigh_dis_ratio * min(dis))
    - graph_x_source: the X source, in ('sct','raw')
    - out_sct: whether to output SCT normed counts
    - out_raw: whether to output raw counts
    '''
    
    # ------- #
    # loading #
    # ------- #
    # accept data format: tsv, txt, h5, h5ad, mtx, AnnData, pd.DataFrame
    if type(st_count_path) == sc.AnnData:
        st_count_obj = st_count_path
    elif type(st_count_path) == str:
        st_count_obj = load_ST_data(st_count_path, load_sep=load_sep, var_names=var_names, transpose=transpose)
    st_count_obj.var_names_make_unique()
    
    if sample_prefix != None:
        st_count_obj.obs_names = ['{}@{}'.format(sample_prefix, tmp_bar) for tmp_bar in st_count_obj.obs_names]

    # don't forget to add prefix in coordinates
    
        
    # ----------- #
    # SCTransform #
    # ----------- #
    # run Seurat SCTransform v2
    st_count_obj = SCTransform_Seurat(st_count_obj,
                                     out_sct=out_sct, out_raw=out_raw, out_h5ad=out_h5ad, save_dir=save_dir)
    
    # ------------ #
    # trans to rxn #
    # ------------ #
    


    # -------------- #
    # build ST graph #
    # -------------- #

    
    return st_count_obj


    
def load_ST_data(st_count_path, load_sep='\t', var_names='gene_symbols', transpose=True):
    '''
    load ST data from provided files,
    accept file format: tsv, txt, h5, h5ad, mtx (mtx,bar,feature dir)
    
    return scanpy obj
    '''
    st_count_path = Path(st_count_path)
    if not st_count_path.exists():
        logging.error('The file not exists')
        raise
    
    if st_count_path.is_dir():
        # load mtx format data, input should be the path to dir of mtx, barcode, feature files.
        st_count_obj = sc.read_10x_mtx(st_count_path, var_names=var_names)
    else:
        tmp_suf = st_count_path.suffix
        if tmp_suf == '.tsv':
            st_count_obj = sc.read_csv(st_count_path, delimiter='\t')
        elif tmp_suf == '.csv':
            st_count_obj = sc.read_csv(st_count_path)
        elif tmp_suf == '.txt':
            st_count_obj = sc.read_csv(st_count_path, delimiter=load_sep)
        elif tmp_suf == '.h5':
            st_count_obj = sc.read_10x_h5(st_count_path)
        elif tmp_suf == '.h5ad':
            st_count_obj = sc.read_h5ad(st_count_path)
        else:
            logging.error('File format not support')
            raise
            
        if tmp_suf not in ['.h5','.h5ad'] and transpose==True:
            # in txt format, rows are often genes & cols are often cells
            st_count_obj = st_count_obj.transpose()
            
    return st_count_obj


def SCTransform_Seurat(st_count_obj, 
                       out_sct=True, out_raw=True, out_h5ad=True, save_dir='./'):
    '''
    run Seurat version of SCTransform (v2)
    need rpy2, anndata2ri (Python); Seurat, glmGamPoi, sctransform (R)

    return: scanpy obj with layers: ['SCT_data', 'SCT_counts']
    '''
    anndata2ri.activate()
    pandas2ri.activate()
    
    seurat = importr('Seurat')
    mat = st_count_obj.X
    
    # trans scanpy obj to seurat obj
    cell_names = st_count_obj.obs_names
    gene_names = st_count_obj.var_names
    r.assign('mat', mat.T)
    r.assign('cell_names', cell_names)
    r.assign('gene_names', gene_names)
    r('colnames(mat) <- cell_names')
    r('rownames(mat) <- gene_names')
    r('seurat_obj <- CreateSeuratObject(mat)')
    
    # SCTransform
    r('seurat_obj <- SCTransform(seurat_obj, verbose=F, vst.flavor="v2")')

    # remove filtered genes during SCT
    filtered_genes = r('rownames(mat)[!rownames(mat) %in% rownames(seurat_obj)]')
    st_count_obj = st_count_obj[:,~np.isin(st_count_obj.var_names, filtered_genes)]
    
    # ----------- #
    # write files #
    # ----------- #
    
    # ST raw counts
    tmp_out = pd.DataFrame(st_count_obj.X).T
    tmp_out.index = st_count_obj.var_names
    tmp_out.columns = st_count_obj.obs_names
    if out_raw == True:
        tmp_out.to_csv(Path(save_dir,'ST_counts.tsv'), sep='\t')
    
    # ST SCT normalized
    tmp_out = pd.DataFrame(r['as.matrix'](r('seurat_obj@assays$SCT@data')))
    tmp_out.index = r('rownames(seurat_obj@assays$SCT@data)')
    tmp_out.columns = r('colnames(seurat_obj@assays$SCT@data)')
    st_count_obj.layers['SCT_data'] = tmp_out.T
    if out_sct == True:
        tmp_out.to_csv(Path(save_dir,'ST_sct_norm.tsv'), sep='\t')
    
    # ST SCT updated counts
    tmp_out = pd.DataFrame(r['as.matrix'](r('seurat_obj@assays$SCT@counts')))
    tmp_out.index = r('rownames(seurat_obj@assays$SCT@counts)')
    tmp_out.columns = r('colnames(seurat_obj@assays$SCT@counts)')
    st_count_obj.layers['SCT_counts'] = tmp_out.T
    if out_sct == True:
        tmp_out.to_csv(Path(save_dir,'ST_sct_counts.tsv'), sep='\t')

    # save scanpy obj
    if out_h5ad == True:
        st_count_obj.write_h5ad(Path('ST_sct_obj.h5ad'))

    return st_count_obj



    
def build_ST_graph(st_count, st_coord,
                   gene_list=None, spot_list=None,
                   nei_dis_ratio=1, n_nei=6,
                   validate=True
                  ):
    '''
    build ST graph according to node spatial distance
    
    - gene_list & spot_list: genes & spots used in build graph, if == None, use index & column of st_exp as avi gene_list & spot_list
    - nei_dis_ratio: define neighbors according to (min_dis * nei_dis_ratio)
    - n_nei: define neighbors by the top n_nei closest nodes. (didn't used in final)
    - validate: whether validate graph
    '''
    if gene_list == None:
        gene_list = list(st_count.index)
    if spot_list == None:
        spot_list = list(st_count.columns)
            
    # cal spot-wise Euclidean distance
    tmp_coord = np.array([[complex(st_coord.iloc[i,0], st_coord.iloc[i,1]) for i in range(st_coord.shape[0])]])
    st_dis_matrix = abs(tmp_coord.T-tmp_coord)

    # fill edge_index
    edge_index = []
    for tmp_xi in range(st_dis_matrix.shape[0]):
        tmp_row = st_dis_matrix[tmp_xi]
        tmp_index = np.where(tmp_row <= np.min(tmp_row[tmp_row>0])*nei_dis_ratio)
        for tmp_yi in tmp_index[0]:
            edge_index.append([tmp_xi,tmp_yi])
            edge_index.append([tmp_yi,tmp_xi])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    st_count = torch.from_numpy(st_count.T.to_numpy()).double()
    st_coord = torch.from_numpy(st_coord.T.to_numpy()).long()
    
    st_graph = Data(x=st_count,
                    edge_index=edge_index.t().contiguous(),
                    coord=st_coord,
                    barcodes=list(st_count.obs_names))
    
    if validate:
        st_graph.validate(raise_on_error=True)
    
    return st_graph
    
    
    
    
    
    
    
    
    
def build_rxn_meta_dic(meta_model_path, rxn_boundary_limit=100,
                       hgcn_db='./data/HGNC_custom_db.tsv',
                       save_flag=False, outdir='./', save_prefix=''):
    '''
    # ---- this func only need to run once ---- #
    
    load metabolic model & build reaction meta info. dict
    input: metabolic model (.xml) & HGCN gene id database
    output: rxn_meta_dic
    '''
    
    # ---------- #
    # load model #
    # ---------- #
    sbmlDocument = libsbml.readSBMLFromFile(meta_model_path)
    sbml_model = sbmlDocument.model
    rxn_list = sbml_model.getListOfReactions()

    hgcn_db = pd.read_csv(hgcn_db, sep='\t', index_col=0) # used to map HGCN ID to symbol
    gr_pattern = re.compile('<html:p>GENE_ASSOCIATION: ([0-9a-zA-Z: ()]*)</html:p>')
    
    rxn_meta_dic = {} # reactant, product & reaction reversible & boundaries & related genes
    # metabolites_dic = {}
    # rxn_metabolites = []
    # all_rxn_ids = []


    for tmp_rxn in rxn_list:
        tmp_rxn_id = tmp_rxn.getId()[2:].replace('_LPAREN_','(').replace('_RPAREN_',')')
        # all_rxn_ids.append(tmp_rxn_id)


        # -------- #
        # rxn type #
        # -------- #
        # tmp_ex_flag = tmp_rxn_id.startswith('EX_')


        # --------------------------------------------------------------------------------------- #
        # just compare whether the react & prod are the same to classify the transport reactions
        # we need to find out a better way later
        # --------------------------------------------------------------------------------------- #


        # tmp_trans_flag = tmp_rxn_id in transport_rxns

        # if tmp_ex_flag:
        #     tmp_rxn_type = 'EX'
        # elif tmp_trans_flag:
        #     tmp_rxn_type = 'TP'
        # else:
        #     tmp_rxn_type = 'RX'

        # if tmp_rxn_id not in rxn_meta_dic:  # do not need this any more, no dupulicated rxn IDs
        rxn_meta_dic.update({
            tmp_rxn_id: {
                # 'rxn_type': tmp_rxn_type
                'rxn_type': 'RX'
            }
        })

        # -------- #
        # reactant #
        # -------- #
        tmp_rea_dic = {}
        for tmp_rea in tmp_rxn.getListOfReactants():
            # tmp_rea_dic 
            # key: reactant name (species)
            # value: Stoichiometry
            tmp_meta_name = tmp_rea.getSpecies()[2:] # trim the prefix of metabolite name
            tmp_rea_dic.update({
                tmp_meta_name: tmp_rea.getStoichiometry() # tmp_rea.getStoichiometry() only have positive value
            })

        rxn_meta_dic[tmp_rxn_id].update({
            'reactants': tmp_rea_dic
        })

        # ------- #
        # product #
        # ------- #
        tmp_prod_dic = {}
        for tmp_prod in tmp_rxn.getListOfProducts():
            # tmp_prod_dic: the same as tmp_rea_dix
            tmp_meta_name = tmp_prod.getSpecies()[2:]
            tmp_prod_dic.update({
                tmp_meta_name: tmp_prod.getStoichiometry()
            })

        rxn_meta_dic[tmp_rxn_id].update({
            'products': tmp_prod_dic
        })

        # -------------- #
        # rxn boundaries #
        # ---------------#
        # the default up and low bound
        for tmp_p in tmp_rxn.getKineticLaw().getListOfParameters():
            if tmp_p.getId() == 'LOWER_BOUND':
                tmp_lb = tmp_p.getValue()
                tmp_lb = -1*rxn_boundary_limit if tmp_lb == -np.inf else tmp_lb
            elif tmp_p.getId() == 'UPPER_BOUND':
                tmp_ub = tmp_p.getValue()
                tmp_ub = rxn_boundary_limit if tmp_ub == np.inf else tmp_ub

        rxn_meta_dic[tmp_rxn_id].update({
            'boundary': {
                'up': tmp_ub,
                'low': tmp_lb
            }
        })

        # ---------- #
        # reversible #
        # ---------- #
        rxn_meta_dic[tmp_rxn_id].update({
            'reversible': tmp_rxn.getReversible()
        })
        
        # ---------------- #
        # gene association #
        # ---------------- #
        tmp_note = tmp_rxn.getNotesString()
        tmp_gid = gr_pattern.search(tmp_note).group(1)
        if len(tmp_gid) == 0:
            # no gene association
            tmp_gsym = []
        else:
            tmp_gid = tmp_gid.replace('(','').replace(')','').replace('and','or').replace('HGNC:HGNC:','HGNC:').split(' or ')
            # there has this kind of note: 'HGNC:30866 or (HGNC:6535 and HGNC:6541) or HGNC:6535'
            # though HGNC:6535 and HGNC:6541 needs to be together, but we only need to use mean or max to cal rxn exp
            # so no diff. for us to just treat them all as 'or'
            # sometimes, notes like 'HGNC:2698 and HGNC:HGNC:987' may also exist
            tmp_gsym = list(hgcn_db.loc[tmp_gid, 'Approved symbol'].values)

        rxn_meta_dic[tmp_rxn_id].update({
            'gene_association': list(set(tmp_gsym))
        })
        
    if save_flag and Path(outdir).exists():
        with open(Path(outdir, save_prefix, '.pkl'), 'wb') as f:
            pkl.dump(rxn_meta_dic, f)
        
    return rxn_meta_dic











    