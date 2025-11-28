#!/usr/bin/python3
# -*- coding:utf-8 -*-
#
# corpus_latent.py: data structure of sparse count matrix with
# a latent topic proportion vector for each nnz element in  
# doc-word count matrix in LDA model
#
# To understand:
# - torch coo sparse tensor
# - tensor advanced indexing
# - index_add_()

import torch


def sparse_vec2matrix(sparse_vec):
    """
    convert a coo sparse vector to matrix
    """
    V, = sparse_vec.size()
    m_indices = torch.zeros(1, sparse_vec._nnz(), dtype=torch.long)  # (1, nnz)
    new_indices = torch.cat([m_indices, sparse_vec.indices()], dim=0)  # (2, nnz)
    return torch.sparse_coo_tensor(new_indices, sparse_vec.values(), size=(1, V)).coalesce()

class CorpusLatentAllocation:
    def __init__(self, mv_count: torch.Tensor, K):
        """
        mv_count: coo sparse vector or sparse matrix

        if mv_count is a coo sparse vector, convert it to sparse matrix
        with size (1, V)
        """
        if len(mv_count.size()) == 1:
            self.mv_count = sparse_vec2matrix(mv_count)
        else:
            self.mv_count = mv_count
        self.nnz, = mv_count.values().shape
        self.K = K

        # self.mvk_phi = None  # (nnz, K)
        self.mvk_phi = torch.full((self.nnz, self.K), 1./self.K)

    def update_phi(self, phi):
        self.mvk_phi = phi  # (nnz, K)

    def pseudo_mk(self):
        pseudo_vk = self.mv_count.values().unsqueeze(1) \
                    * self.mvk_phi  # (nnz, K)
        M = self.mv_count.size()[0]
        m_indices = self.mv_count.indices()[0]  # (nnz,)
        mk_pseudo = torch.zeros((M, self.K))  # (M, K)
        return mk_pseudo.index_add_(0, m_indices, pseudo_vk)  # (M, K)

    def pseudo_kv(self):
        pseudo_vk = self.mv_count.values().unsqueeze(1) \
                    * self.mvk_phi  # (nnz, K)
        V = self.mv_count.size()[-1]
        base = torch.zeros((self.K, V))  # (K, V)
        v_indices = self.mv_count.indices()[-1]  # (nnz,)

        return base.index_add_(1, v_indices, pseudo_vk.T)  # (K, V)
            

def main():
    pass

if __name__ == "__main__":
    main()
