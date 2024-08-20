from torch.nn import Sequential,  ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool, dense_diff_pool, DenseGINConv, GPSConv
from torch_geometric.nn.models import MLP, AttentiveFP
from torch_geometric.utils import remove_self_loops
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_scatter import scatter_mean
from layer_het import *
import sklearn.covariance
import numpy as np
from  torch.autograd import Variable
from torch import autograd
        
from torch.nn import (
    BatchNorm1d,
    Embedding,
    ModuleList,
    ReLU,
    Sequential,
)     
from typing import Any, Dict, Optional
from torch_geometric.nn import Linear

# SSL
import GCL.losses as L
from GCL.models import DualBranchContrast

class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()
        full_bond_feature_dims = [22, 6, 2]
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   




class RingEncoder(torch.nn.Module):

    def __init__(self, emb_dim, pe=False):
        super(RingEncoder, self).__init__()
        
        self.ring_embedding_list = torch.nn.ModuleList()
        full_ring_feature_dims = [60]
        for i, dim in enumerate(full_ring_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.ring_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.ring_embedding_list[i](x[:,i])

        return x_embedding
    
    
class RingBondDegreeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_edge_types=17):
        super(RingBondDegreeEncoder, self).__init__()
        
        self.ring_embedding_list = torch.nn.ModuleList()
        full_ring_feature_dims = [7]*num_edge_types
        for i, dim in enumerate(full_ring_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            emb.weight.data[0] = 0.0
            self.ring_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.ring_embedding_list[i](x[:,i])

        return x_embedding    

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            Linear(in_size, hidden_size),
            nn.Tanh(),
            Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1).squeeze(), beta



class HeteroTransformer(nn.Module):
    def __init__(self, metadata,
                 nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 attn_dropout=0.0,
                 norm=None, 
                 transformer_norm=None,
                 heads=4,
                 pool='add',
                 conv='GINE',
                 inter_conv='GINE',
                 ring_conv='GINE',
                 jk = 'cat',
                 final_jk = 'cat',
                 intra_jk = 'cat',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=False,
                 target_task = None,
                 ring_init = 'atom_deepset',
                 mol_init = 'atom_deepset',
                 pair_init = 'random',
                 pe_dim = 0,
                 pe_emb_dim = 128,
                 num_lin_layer = 1,
                 model = 'Het',
                 contrastive = False,
                 num_deepset_layer = 1,
                 init_embs=False,
                 padding=True,
                 mask_non_edge=False,
                 cat_pe = False,
                 use_bias = False,
                 add_mol = False,
                 combine_mol='add',
                 float_pe = False,
                 combine_edge='add',
                 root_weight=True,
                 num_ring_edge_types=1,
                 clip_attn=False,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.ring_init = ring_init
        self.mol_init = mol_init
        self.contrastive = contrastive
        self.cat_pe = cat_pe
        self.add_mol = add_mol
        self.float_pe = float_pe
        self.num_ring_edge_types = num_ring_edge_types
            
            
        first_residual = True
        Encoder =  Het_Transfomer 

        self.encoder = Encoder(metadata, dim=nhid, gnn=conv, inter_gnn=inter_conv, ring_gnn=ring_conv, 
                               num_gc_layers=nlayer, heads=heads, norm=norm, transformer_norm=transformer_norm, dropout=dropout, attn_dropout=attn_dropout, pool = pool,
                               aggr=aggr, jk = jk, intra_jk=intra_jk, first_residual=first_residual, init_embs=init_embs, padding=padding, mask_non_edge=mask_non_edge, 
                               residual=residual, use_bias=use_bias, add_mol=add_mol, combine_mol=combine_mol, root_weight=root_weight, combine_edge=combine_edge, clip_attn=clip_attn)
        self.atom_encoder = AtomEncoder(nhid) 
        self.ring_encoder = RingEncoder(nhid-pe_emb_dim) if cat_pe else  RingEncoder(nhid) # TODO: ring type + PE
    
        # Edge attr encoder
        ring_bond_encoder = torch.nn.Embedding(42,  nhid) # 40 for self loop, 41 for virtual edge
        torch.nn.init.xavier_uniform_(ring_bond_encoder.weight.data)   
        ar_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ar_bond_encoder.weight.data)  
        ra_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ra_bond_encoder.weight.data)  
          
        self.bond_encoder = nn.ModuleDict({'a2a': BondEncoder(nhid), 'a2r': ar_bond_encoder, 
                                           'r2r': ring_bond_encoder, 'r2a': ra_bond_encoder,
                                           })
        if self.add_mol:
            self.mol_encoder = torch.nn.Embedding(2,  nhid-pe_emb_dim) if cat_pe else torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(self.mol_encoder.weight.data)   

                    
        if pe_dim > 0:
            if float_pe:
                self.pe_encoder = nn.Sequential( Linear(pe_dim, pe_emb_dim)) if cat_pe else  nn.Sequential(Linear(pe_dim, nhid))
            else:
                if num_ring_edge_types == 1:
                    self.pe_encoder = torch.nn.Embedding(pe_dim+1,  pe_emb_dim, padding_idx=0) if cat_pe else torch.nn.Embedding(pe_dim+1,  nhid, padding_idx=0)
                    torch.nn.init.xavier_uniform_(self.pe_encoder.weight.data)       
                    self.pe_encoder.weight.data[0] = 0.0    
                else:
                    self.pe_encoder = RingBondDegreeEncoder(pe_emb_dim, num_ring_edge_types) if cat_pe else RingBondDegreeEncoder(nhid, num_ring_edge_types) 
        if ring_init.startswith('atom_deepset') or ring_init == 'deepset_random':
            # if ring_init[-1] == '1':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, nhid), nn.Dropout(dropout))
            # elif ring_init[-1] == '2':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout))
            # elif ring_init[-1] == '3':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), nn.Dropout(dropout), Linear(2*nhid, nhid), ReLU(), nn.Dropout(dropout))               
            # elif ring_init[-1] == '4':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, nhid))
            # elif ring_init[-1] == '5':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(nhid))      
            # elif ring_init[-1] == '6':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.LayerNorm(nhid)) 
            # elif ring_init[-1] == '7':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), Linear(2*nhid, nhid))         
            # elif ring_init[-1] == '8':
            #     self.ring_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), Linear(nhid, nhid), nn.Dropout(dropout))                                                                   
            # else:
            self.ring_deepset = nn.Sequential(Linear(nhid, nhid-pe_emb_dim), ReLU()) if cat_pe else nn.Sequential(Linear(nhid, nhid), ReLU())
            
        penultimate_dim = (nlayer+1)*nhid  if jk == 'cat' else nhid
        if final_jk == 'cat':
            final_dim = penultimate_dim * 2
            if combine_mol == 'cat':
                final_dim = final_dim + penultimate_dim
        else:
            final_dim = penultimate_dim
        self.lin = Linear(final_dim, nclass) if num_lin_layer == 1 else nn.Sequential(Linear(final_dim, penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(penultimate_dim, nclass))
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        

            
    def forward(self, data):
        device = data.y.device
        # Initialize node embeddings
        # Atom
        x_atom = self.atom_encoder(data.x_dict['atom'].int())
        # Ring
        if self.ring_init == 'random':
            x_ring = self.ring_encoder(data.x_dict['ring'].int())
        elif self.ring_init == 'zero':
            x_ring = torch.zeros((data['ring'].ptr[-1].item(), x_atom.shape[1])).to(device)
        elif self.ring_init.startswith('atom_deepset') or self.ring_init == 'deepset_random':
            ringatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_ringatoms)]
            ringatoms_batch = torch.cat(ringatoms_batch, dim=0) # Mark each ringatom belongs to which graph
            ringatoms_ptr = data['atom'].ptr[ringatoms_batch] # Get the pointer of each ringatom in the global graph
            ringatoms = data.ring_atoms + ringatoms_ptr # Get the index of each ringatom in the global graph         
            ring_atoms_map = data.ring_atoms_map + data['ring'].ptr[ringatoms_batch] 
            x_ring = global_add_pool(x_atom[ringatoms], ring_atoms_map) # motif embeddings: pool corresponding node embedding
            x_ring = F.dropout(self.ring_deepset(x_ring), p=self.dropout, training=self.training)
            if self.add_mol:
                x_ring = torch.cat((x_ring, self.mol_encoder(torch.LongTensor([0]).to(device))), 0)
        elif self.ring_init == 'add' or self.ring_init == 'mean':
            ringatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_ringatoms)]
            ringatoms_batch = torch.cat(ringatoms_batch, dim=0) # Mark each ringatom belongs to which graph
            ringatoms_ptr = data['atom'].ptr[ringatoms_batch] # Get the pointer of each ringatom in the global graph
            ringatoms = data.ring_atoms + ringatoms_ptr # Get the index of each ringatom in the global graph         
            ring_atoms_map = data.ring_atoms_map + data['ring'].ptr[ringatoms_batch] 
            if self.ring_init == 'add':
                x_ring = global_add_pool(x_atom[ringatoms], ring_atoms_map) 
            elif self.ring_init == 'mean':
                x_ring = global_mean_pool(x_atom[ringatoms], ring_atoms_map)
        else:
            raise NameError(f"{self.ring_init} is not implemented!")
        if self.pe_dim > 0: # TODO: delete in CEPDB
            if self.cat_pe:
                if not self.float_pe:
                    if self.num_ring_edge_types == 1:
                        x_ring = torch.cat((x_ring, self.pe_encoder(data['ring'].ring_pe.reshape(-1).int())), -1)

                    else:
                        x_ring = torch.cat((x_ring, self.pe_encoder(data['ring'].ring_pe.int())), -1)
                else:
                    x_ring = torch.cat((x_ring, self.pe_encoder(data['ring'].ring_pe)), -1)
            else:
                if not self.float_pe:
                    if self.num_ring_edge_types == 1:
                        x_ring = x_ring + self.pe_encoder(data['ring'].ring_pe.reshape(-1).int())
                    else:
                        x_ring = x_ring + self.pe_encoder(data['ring'].ring_pe.int())
                else:
                    x_ring = x_ring + self.pe_encoder(data['ring'].ring_pe)
        # Encode graph
        


        x_dict = {'atom': x_atom, 'ring':  x_ring}
        edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr) for edge_type, edge_attr in data.edge_attr_dict.items() }
        
        
        atom_embs, ring_embs, pair_embs, mol_embs = self.encoder(x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict, edge_type_dict=data.edge_attr_dict, data=data)

            # ring_embs, mol_embs = ring_embs[:-1], ring_embs[-1]  # Remove virtual node
        
        return atom_embs, ring_embs, pair_embs, mol_embs
    def get_embs(self, data):
        atom_embs, ring_embs, pair_embs, mol_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, ring_embs], dim=1)
        elif self.final_jk == 'add': # TODO: delete in CEPDB
            graph_embs = atom_embs + ring_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, ring_embs]       
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, ring_embs]
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'ring':
            graph_embs = ring_embs
        elif self.final_jk == 'mol':
            graph_embs = mol_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")   
        return graph_embs                
    def predict_score(self, data):
        atom_embs, ring_embs, pair_embs, mol_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, ring_embs], dim=1)
        elif self.final_jk == 'add': # TODO: delete in CEPDB
            graph_embs = atom_embs + ring_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, ring_embs]       
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, ring_embs]
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'ring':
            graph_embs = ring_embs
        elif self.final_jk == 'mol':
            graph_embs = mol_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")        

        scores = self.lin(graph_embs)    
        
        return scores

    def calc_contra_loss(self, data):
        atom_embs, ring_embs, pair_embs, mol_embs = self(data)
        g1, g2 = [self.project(g) for g in [atom_embs, ring_embs]]
        loss = self.ssl_criterion(g1=g1, g2=g2)
        return loss
    
    def calc_loss(self, data):
        
        device = data.y.device
        scores = self.predict_score(data)

        mask = (data.y != 0).float().to(device) # TODO: delete in CEPDB
        scores = scores * mask
        # y = torch.nan_to_num(data.y, nan=0.0).to(device)
        loss = self.criterion(scores, data.y)

        return loss
