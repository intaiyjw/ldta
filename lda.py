#!/usr/bin/python3
# -*- coding:utf-8 -*-

from copy import deepcopy
from typing import Union, Callable

import torch
from torch import logsumexp

from tensor_mat import tense_array_torch as expand_tensor
from corpus_latent import CorpusLatentAllocation as Latent
from dists import Dirichlet as Dir, Beta_Liouville as BL, Generalized_Dirichlet as GDir
from message_passing import mean_u2dist as mean_u2dist

ALLOWED_PRIORS = [Dir, BL, GDir]


def get_elbo_m(model: "TopicModel", corpus: "CorpusSufficient"):
    # elbo_1: 2, 3, 5
    m_indices = corpus.latent.mv_count.indices()[0]  # (nnz,)
    v_indices = corpus.latent.mv_count.indices()[-1]  # (nnz,)

    mean_log_theta = corpus.post_zetas.mean_log_theta()[m_indices, :]  # (nnz, K)
    log_corpusphi = model.log_corpus_phi[:, v_indices]  # (K, nnz)

    weights = log_corpusphi.T + mean_log_theta - torch.log(corpus.latent.mvk_phi)  # (nnz, K)
    elbo_nnz = torch.sum(corpus.latent.mvk_phi * weights, dim=-1)  # (nnz,)
    elbo_nnz *= corpus.latent.mv_count.values()  # (nnz,)

    elbo_1 = torch.zeros((corpus.M,))  # notice datatype
    elbo_1.index_add_(0, m_indices, elbo_nnz)  # (M,)
    # elbo_2:
    prior_xis = model.prior_cls(expand_tensor(model.prior.para, 0, corpus.M))

    elbo_2 = torch.sum((prior_xis.nat_para() - 
                        corpus.post_zetas.nat_para()) * corpus.post_zetas.mean_u(), 
                        dim=tuple(range(-model.prior.KernelOrder, 0)))  # (M,)
    # elbo_3:
    elbo_3 = corpus.post_zetas.log_normalizer() - prior_xis.log_normalizer()  # (M,)

    return elbo_1 + elbo_2 + elbo_3


class TopicModel():
    def __init__(self, K: int, V: int, prior_cls: Callable = Dir, xi = 1.0):
        self.K = K
        self.V = V

        if prior_cls not in ALLOWED_PRIORS:
            raise ValueError("Invalid prior class passed")
        else:
            self.prior_cls = prior_cls
            self.prior = prior_cls.create(
                prior_cls.create_para(K = K, val = xi)
            )

        self.log_corpus_phi = self._random_init_log_corpus_phi(K, V)

    @staticmethod
    def _random_init_log_corpus_phi(K: int, V: int):
        corpus_phi = torch.full((K, V), 1./V) + \
                     torch.rand(K, V)
        log_corpus_phi = torch.log(corpus_phi) - \
                         torch.log(torch.sum(corpus_phi, dim=-1, keepdim=True))
        return log_corpus_phi
    
    def max_elbo_update_log_corpus_phi(self, corpus: "CorpusSufficient"):
        pseudo_kv = corpus.latent.pseudo_kv()  # (K, V)
        normalizer = torch.sum(pseudo_kv, dim=-1, keepdim=True)  # (K, 1)
        self.log_corpus_phi = torch.log(pseudo_kv) - torch.log(normalizer)

    def max_elbo_update_prior(self, corpus: "CorpusSufficient"):
        mean_u_to_match = corpus.post_zetas.mean_u().sum(dim=0) / corpus.M  # (D,)
        self.prior = mean_u2dist(mean_u_to_match, dist_cls=self.prior_cls)


class CorpusSufficient():
    def __init__(self, mv_count: torch.Tensor, init_model: "TopicModel"):
        """
        mv_count: coo sparse matrix or sparse vector
        init_model: to get K the number of topics, and type of prior
        """
        if mv_count.size()[-1] != init_model.V:
            raise ValueError("Corpus and Model V not matched")

        self.latent = Latent(mv_count, init_model.K)
        self.M, self.V = self.latent.mv_count.size()
 
        self.var_cls = init_model.prior_cls
        self.post_zetas = None

        # optional for ep
        self.log_s_v = None  # (nnz, )

    def get_perplexity(self, log_corpus_phi):
        m_indices = self.latent.mv_count.indices()[0]  # (nnz,)
        v_indices = self.latent.mv_count.indices()[-1]  # (nnz,)
        n_mv = self.latent.mv_count.values()  # (nnz,)

        mean_theta = self.post_zetas.mean_theta()  # (M, K)
        if (mean_theta == 0).any():
            mean_theta = torch.clamp(mean_theta, min=1e-323)
            mean_theta = mean_theta / mean_theta.sum(dim=-1, keepdim=True)
        log_mean_theta = torch.log(mean_theta)  # (M, K)

        loglikelihood_mv = logsumexp(log_mean_theta[m_indices, :] + \
                                     log_corpus_phi[:, v_indices].T, dim=-1)  # (nnz, )
        
        perplex = torch.exp(- (n_mv * loglikelihood_mv).sum() / n_mv.sum())  # scalar tensor
        return perplex
        
    def update_post_zetas(self, prior):
        """
        Based on:
        - self.latent
        - prior from instance TopicModel
        """
        self.post_zetas = self.var_cls(expand_tensor(prior.para, 0, self.M))
        self.post_zetas.update_posterior(self.latent.pseudo_mk())
    
    # variational inference

    def update_latent(self, log_corpus_phi):
        """
        Based on:
        - self.post_zetas
        - log_corpus_phi from instance TopicModel
        """
        m_indices = self.latent.mv_count.indices()[0]
        v_indices = self.latent.mv_count.indices()[-1]
        mean_log_theta = self.post_zetas.mean_log_theta()[m_indices, :]  # (nnz, K)
        log_corpus_phi = log_corpus_phi[:, v_indices]  # (K, nnz)

        log_latent = mean_log_theta + log_corpus_phi.T  # (nnz, K)
        log_normalizer = logsumexp(log_latent, dim=-1, keepdim=True)  # (nnz, 1)

        ##
        new_phi = torch.exp(log_latent - log_normalizer)
        if (new_phi == 0).any():
            new_phi = torch.clamp(new_phi, min=1e-323)
            new_phi = new_phi / new_phi.sum(dim=-1, keepdim=True)
        self.latent.update_phi(new_phi)
        ##

        # self.latent.update_phi(torch.exp(log_latent - log_normalizer))

    def var_infer(self, model: "TopicModel", max_iter=50, stop_iter=1e-6):
        if model.K != self.latent.K:
            raise ValueError("Corpus and Model K not matched")
        K = self.latent.K
        nnz = self.latent.nnz

        self.latent.update_phi(torch.full((nnz, K), 1./K))  # notice datatype

        self.update_post_zetas(model.prior)
        self.update_latent(model.log_corpus_phi)
        elbo_m = get_elbo_m(model, self)  # (M,)

        converged = 1  # change rate of elbo of a document
        iter_count = 0
        while ((converged > stop_iter) and (iter_count < max_iter or max_iter == -1)):
            iter_count += 1
            elbo_old = elbo_m

            # coordinate ascend
            self.update_post_zetas(model.prior)
            self.update_latent(model.log_corpus_phi)
            elbo_m = get_elbo_m(model, self)

            converged = torch.max((elbo_old - elbo_m) /  # elbo will not descend, abs() not needed
                                  elbo_old).item()
            
            
    # Expectation Propagation

    def log_likelihood_ep(self, prior):
        """
        based on:
        - self.log_s_v
        - self.latent.mv_count
        - self.post_zetas.log_normalizer()
        - self.var_cls
        - prior from instance TopicModel
        """
        m_indices = self.latent.mv_count.indices()[0]
        log_s_m = torch.zeros((self.M, ))  # (M, )
        log_s_m.index_add_(0, m_indices, 
                           self.latent.mv_count.values() * self.log_s_v)
        return self.post_zetas.log_normalizer() \
               - self.var_cls(expand_tensor(prior.para, 0, self.M)).log_normalizer() \
               + log_s_m  # (M, )
    
    def update_latent_ep_original(self, log_corpus_phi, eps=1e-8):
        """
        Based on:
        - self.post_zetas
        - self.latent.mvk_phi
        - log_corpus_phi from an instance TopicModel
        """
        m_indices = self.latent.mv_count.indices()[0]
        v_indices = self.latent.mv_count.indices()[-1]
        # cavity dists
        cavity_dists = self.var_cls(self.post_zetas.para[m_indices])  # (nnz, D)
        cavity_dists.update_posterior(-self.latent.mvk_phi)  # (nnz, D)
        # log_s_v
        log_weights = log_corpus_phi.T[v_indices] + torch.log(cavity_dists.mean_theta())  # (nnz, K)
        log_sum_weights = logsumexp(log_weights, dim=-1)  # (nnz, ), log_z_v

        self.log_s_v = cavity_dists.log_normalizer() - self.post_zetas.log_normalizer()[m_indices] \
                       + log_sum_weights  # (nnz, )

        # mean_u of tilted dists
        sum_weights = torch.exp(log_sum_weights.reshape(log_sum_weights.shape + ((1,) * cavity_dists.KernelOrder)))  # (nnz, KernelOrder * (1,))
        weights = torch.exp(log_weights.reshape(log_weights.shape + ((1,) * cavity_dists.KernelOrder)))  # (nnz, K, KernelOrder * (1,))

        mean_u_shifted = cavity_dists.mean_u_shifted()  # (nnz, K, D)

        tilted_mean_u = torch.sum(weights * mean_u_shifted, dim=1) / sum_weights  # (nnz, D)
        # message passing
        sep_dists = mean_u2dist(tilted_mean_u, dist_cls=self.var_cls)  # (nnz, D)
        # update of self.latent.mvk_phi
        new_mvk_phi = torch.clamp(sep_dists.bayes_minus(cavity_dists), 
                                  min=eps)  # (nnz, K) # ?
        new_mvk_phi = new_mvk_phi / new_mvk_phi.sum(dim=-1, keepdim=True)
        self.latent.update_phi(new_mvk_phi)

    def update_latent_ep(self, log_corpus_phi, eps=1e-8, batch_size=100_000):
        """
        Memory-safe EP latent update using mini-batching.
        """

        m_indices = self.latent.mv_count.indices()[0]
        v_indices = self.latent.mv_count.indices()[-1]
        nnz = m_indices.size(0)
        K = self.latent.K

        # Pre-allocate final tensors
        new_log_s_v = torch.empty((nnz,), device=log_corpus_phi.device)
        new_mvk_phi = torch.empty((nnz, K), device=log_corpus_phi.device)

        for i in range(0, nnz, batch_size):
            # Slice batch
            b_slice = slice(i, min(i + batch_size, nnz))
            b_m_idx = m_indices[b_slice]
            b_v_idx = v_indices[b_slice]

            # Step 1: Cavity distributions
            cavity_dists = self.var_cls(self.post_zetas.para[b_m_idx])  # (B, D)
            cavity_dists.update_posterior(-self.latent.mvk_phi[b_slice])  # (B, D)

            # Step 2: Compute log_s_v
            log_weights = log_corpus_phi[:, b_v_idx].T + torch.log(cavity_dists.mean_theta())  # (B, K)
            log_sum_weights = logsumexp(log_weights, dim=-1)  # (B,)

            new_log_s_v[b_slice] = (
                cavity_dists.log_normalizer()
                - self.post_zetas.log_normalizer()[b_m_idx]
                + log_sum_weights
            )

            # Step 3: Compute tilted mean_u
            reshaped_log_sum = log_sum_weights.reshape(-1, *[1]*cavity_dists.KernelOrder)
            sum_weights = torch.exp(reshaped_log_sum)  # (B, 1) or (B, 1, ...) for broadcasting

            reshaped_weights = torch.exp(log_weights).reshape(
                log_weights.shape + (1,) * cavity_dists.KernelOrder
            )  # (B, K, D)

            mean_u_shifted = cavity_dists.mean_u_shifted()  # (B, K, D)
            tilted_mean_u = torch.sum(reshaped_weights * mean_u_shifted, dim=1) / sum_weights  # (B, D)

            # Step 4: Compute Bayes minus (update messages)
            sep_dists = mean_u2dist(tilted_mean_u, dist_cls=self.var_cls)
            mvk_phi_batch = torch.clamp(sep_dists.bayes_minus(cavity_dists), min=eps)  # (B, K)
            mvk_phi_batch /= mvk_phi_batch.sum(dim=-1, keepdim=True)

            new_mvk_phi[b_slice] = mvk_phi_batch

            # Free unused memory
            del (
                cavity_dists, log_weights, log_sum_weights, reshaped_weights,
                mean_u_shifted, tilted_mean_u, sep_dists, mvk_phi_batch, sum_weights
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Store results
        self.log_s_v = new_log_s_v
        self.latent.update_phi(new_mvk_phi)

    def var_infer_ep(self, model: "TopicModel", max_iter=10, stop_iter=1e-4):
        if model.K != self.latent.K:
            raise ValueError("Corpus and Model K not matched")
        K = self.latent.K
        nnz = self.latent.nnz

        self.latent.update_phi(torch.full((nnz, K), 1./K))

        self.update_post_zetas(model.prior)
        self.update_latent(model.log_corpus_phi)
        self.update_post_zetas(model.prior)
        self.update_latent_ep(model.log_corpus_phi)
        log_likelihood_m = self.log_likelihood_ep(model.prior)  # (M, )
        elbo = get_elbo_m(model, self)  # (M, )

        self.ep_loglikelihood = []
        self.ep_loglikelihood.append(log_likelihood_m.sum().item())
        self.ep_elbo = []
        self.ep_elbo.append(elbo.sum().item())

        converged = 1
        iter_count = 0
        while ((converged > stop_iter) and (iter_count < max_iter or max_iter == -1)):
            iter_count += 1
            log_likelihood_old = log_likelihood_m
            elbo_old = elbo

            self.update_post_zetas(model.prior)
            self.update_latent_ep(model.log_corpus_phi)

            log_likelihood_m = self.log_likelihood_ep(model.prior)  # (M, )
            elbo = get_elbo_m(model, self)  # (M, )
            self.ep_loglikelihood.append(log_likelihood_m.sum().item())
            self.ep_elbo.append(elbo.sum().item())

            converged = torch.max(torch.abs((log_likelihood_m - log_likelihood_old) /
                                            log_likelihood_old)).item()
            
            print(f"** ep iteration {iter_count} **")
            print(f"converged = {converged}")


class LDA():
    def __init__(self, corpus: torch.Tensor, K: int,
                 prior_cls: Callable = Dir, xi = 1.0):
        """
        corpus: sparse matrix
        """
        self.M, self.V = corpus.size()
        self.K = K

        self.model = TopicModel(self.K, self.V, prior_cls, xi)
        self.sufficient = CorpusSufficient(corpus, self.model)
        self.sufficient.var_infer(self.model)

        self.elbo_itr = []

    def fit(self, em_stop_itr=1e-4, em_max_iter=100, est_prior=False):
        elbo = torch.sum(get_elbo_m(self.model, self.sufficient)).item()
        self.elbo_itr.append(elbo)

        em_converged = 1
        em_itr_count = 0
        while ( ((em_itr_count <= 3) or (em_converged > em_stop_itr) or (em_converged < 0))
                and (em_itr_count < em_max_iter) ):
            em_itr_count += 1
            elbo_old = elbo

            print(f"**** em iteration {em_itr_count} ****")

            # M-step:
            self.model.max_elbo_update_log_corpus_phi(self.sufficient)
            if est_prior:
                self.model.max_elbo_update_prior(self.sufficient)
            # E-step:
            self.sufficient.var_infer(self.model)
            # elbo:
            elbo = torch.sum(get_elbo_m(self.model, self.sufficient)).item()
            self.elbo_itr.append(elbo)

            em_converged = (elbo_old - elbo) / elbo_old
            if em_converged < 0:
                em_max_iter *= 2
                print(f"** converged negative, max_iter doubled = {em_max_iter} **")

            print(f"elbo = {elbo};    converged = {em_converged}")
            print("")

    def fit_ep(self, em_stop_itr=1e-4, em_max_iter=100, est_prior=False):
        elbo = torch.sum(get_elbo_m(self.model, self.sufficient)).item()
        self.elbo_itr.append(elbo)

        em_converged = 1
        em_itr_count = 0
        while ( ((em_itr_count <= 3) or (em_converged > em_stop_itr) or (em_converged < 0))
                and (em_itr_count < em_max_iter) ):
            em_itr_count += 1
            elbo_old = elbo

            print(f"**** ep-m iteration {em_itr_count} ****")

            # M-step:
            self.model.max_elbo_update_log_corpus_phi(self.sufficient)
            if est_prior:
                self.model.max_elbo_update_prior(self.sufficient)
            # E-step:
            self.sufficient.var_infer_ep(self.model)
            # elbo:
            elbo = torch.sum(get_elbo_m(self.model, self.sufficient)).item()
            self.elbo_itr.append(elbo)

            em_converged = (elbo_old - elbo) / elbo_old
            if em_converged < 0:
                em_max_iter *= 2
                print(f"** converged negative, max_iter doubled = {em_max_iter} **")

            print(f"elbo = {elbo};    converged = {em_converged}")
            print("")

    def var_infer(self, sparse_doc: torch.Tensor):
        """
        sparse_doc: coo sparse matrix or tensor
        """
        doc_suff = CorpusSufficient(sparse_doc, self.model)
        doc_suff.var_infer(self.model)
        return doc_suff
    
    def var_infer_ep(self, sparse_doc: torch.Tensor):
        """
        sparse_doc: coo sparse matrix or tensor
        """
        doc_suff = CorpusSufficient(sparse_doc, self.model)
        doc_suff.var_infer_ep(self.model)
        return doc_suff



def main():
    pass

if __name__ == "__main__":
    main()
