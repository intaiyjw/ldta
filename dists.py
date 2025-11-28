#!/usr/bin/python3
# -*- coding:utf-8 -*-

import torch
from torch.special import gammaln, psi

from tensor_mat import tense_array_torch as expand_tensor, \
                       vectorized_identity_torch as identity_tensor, \
                       generalized_permute_torch as permute_tensor

from typing import Optional, Union, Tuple, Type


class Exponential_Family:
    KernelOrder = None  # to be overridden

    def __init__(self, para: torch.Tensor):
        # self.para = para.float()
        self.para = para  # notice datatype

        self.BatchOrder = len(para.shape) - self.KernelOrder

        self.KernelShape = para.shape[-self.KernelOrder:]
        self.BatchShape = para.shape[0:-self.KernelOrder]

        self.D = self._get_D()
        self.K = self._get_K()

    def _get_D(self) -> Union[tuple, torch.Size]:
        """
        to be overridden
        """
        pass

    def _get_K(self) -> Union[tuple, torch.Size]:
        """
        to be overridden
        """
        pass

    def nat_para(self) -> torch.Tensor:
        """
        to be overridden

        return: (BatchShape, D)
        """
        pass

    def mean_u(self) -> torch.Tensor:
        """
        return: (BatchShape, D)
        """
        return self._mean_u(self.para)

    def log_normalizer(self) -> torch.Tensor:
        """
        to be overridden

        return: (BatchShape,)
        """
        pass

    def _mean_u(self, para: torch.Tensor) -> torch.Tensor:
        """
        to be overridden

        - para: (..., KernelShape)
        
        return: (..., D)
        """
        pass

    @classmethod
    def create(cls, para: torch.Tensor):
        """
        create an instance of this class
        """
        return cls(para)
    

class Multinomial_Conjugator(Exponential_Family):
    def update_posterior(self, n: torch.Tensor):
        """
        to be overridden

        n: (BatchShape, K)
        """
        # self.para = self._bayes_add(self.para, n.float())
        self.para = self._bayes_add(self.para, n)  # notice datatype

    def mean_log_theta(self) -> torch.Tensor:
        """
        to be overridden

        return: (BatchShape, K)
        """
        pass

    def mean_theta(self) -> torch.Tensor:
        """
        to be overridden

        return: (BatchShape, K)
        """
        pass

    def shifted_para(self) -> torch.Tensor:
        """
        return: (BatchShape, K, KernelShape)
        """
        batch_para = expand_tensor(self.para, self.BatchOrder, self.K)  # (BatchShape, K, KernelShape)
        diag_ones = expand_tensor(identity_tensor(self.K, 1),
                                  0,
                                  self.BatchShape)  # (BatchShape, K, K)
        shifted_para = self._bayes_add(batch_para, diag_ones)  # (BatchShape, K, KernelShape)

        return shifted_para
    
    def mean_u_shifted(self) -> torch.Tensor:
        """
        return expectation of sufficient statistics of shifted para
        - (BatchShape, K, D)
        """
        return self._mean_u(self.shifted_para())  # (BatchShape, K, D)
    
    def mean_theta_derived(self) -> torch.Tensor:
        """
        depends on:
        - self.mean_u()
        - self.mean_theta()
        - self.mean_u_shifted()

        return the expectation matrix of derived dists
        - (BatchShape, D, K)
        """
        batch_mean_u = expand_tensor(self.mean_u(), -1, self.K)  # (BatchShape, D, K)
        batch_mean_theta = expand_tensor(self.mean_theta(), self.BatchOrder, self.D)  # (BatchShape, D, K)
        batch_mean_u_shifted = permute_tensor(self.mean_u_shifted(),
                                              (self.BatchOrder, len(self.K), len(self.D)),
                                              (0, 2, 1))  # (shellShape, D, K)
        mean_theta_derived = batch_mean_theta * batch_mean_u_shifted / batch_mean_u

        return mean_theta_derived

    def _bayes_add(self, para: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        to be overridden

        - para: (..., KernelShape)
        - n: (..., K)

        return: (..., KernelShape)
        """
        pass

    def bayes_minus(self, dist: "Multinomial_Conjugator"):
        """
        to be overridden

        - dist: an instance of the same class, 
                same shape of para (..., KernelShape)

        return: n s.t. dist.update_posterior(n) is self
        """
        pass


class Dirichlet(Multinomial_Conjugator):
    """
    alpha: (alpha_1, alpha_2, ..., alpha_K)
    """
    KernelOrder = 1

    def _get_D(self):
        return self.KernelShape
    
    def _get_K(self):
        return self.KernelShape
    
    def nat_para(self):
        return self.para - 1
    
    def _mean_u(self, para):
        sum_k = torch.sum(para, dim=-1, keepdim=True)  # ...k -> ...1
        mean_u = psi(para) - psi(sum_k)
        return mean_u

    def log_normalizer(self):
        return torch.sum(gammaln(self.para), dim=-1) - \
               gammaln(torch.sum(self.para, dim=-1))
    
    def mean_log_theta(self):
        return self.mean_u()
    
    def mean_theta(self):
        sum_k = torch.sum(self.para, dim=-1, keepdim=True)
        return self.para / sum_k
    
    def _bayes_add(self, para, n):
        return para + n
    
    def bayes_minus(self, dist):
        return self.para - dist.para
    
    @staticmethod
    def create_para(K: int, val: float = 1.0, batch_grid_dim = tuple()) -> torch.Tensor:
        size = batch_grid_dim + (K, )
        para = torch.full(size, val)
        return para


class Beta_Liouville(Multinomial_Conjugator):
    """
    xi: (alpha_1, alpha_2, ..., alpha_K-1, alpha, beta)
    """
    KernelOrder = 1

    def _get_D(self):
        return self.KernelShape
    
    def _get_K(self):
        return tuple(i-1 for i in self.KernelShape)
    
    def nat_para(self):
        nat = self.para - 1
        nat[..., -2] -= self.K[0] - 2
        return nat

    def _mean_u(self, para):
        D, = self.D
        size_alpha_k = D - 2

        sum_alpha_k = torch.sum(para[..., :-2], dim=-1)
        sum_alpha_beta = torch.sum(para[..., -2:], dim=-1)

        batch_sum_alpha_k = expand_tensor(sum_alpha_k, -1, size_alpha_k)
        batch_sum_alpha_beta = expand_tensor(sum_alpha_beta, -1, 2)

        batch_sum = torch.cat((batch_sum_alpha_k, batch_sum_alpha_beta), dim=-1)

        return psi(para) - psi(batch_sum)

    def log_normalizer(self):
        sum_alpha_k = torch.sum(self.para[..., :-2], dim=-1)
        sum_alpha_beta = torch.sum(self.para[..., -2:], dim=-1)
        return torch.sum(gammaln(self.para), dim=-1) \
               - gammaln(sum_alpha_k) - gammaln(sum_alpha_beta)

    def mean_log_theta(self):
        mean_u = self.mean_u()

        D, = self.D
        size_alpha_k = D - 2

        u_alpha = mean_u[..., -2]  # (...,)
        batch_u_alpha = expand_tensor(u_alpha, -1, size_alpha_k)  # (..., D-2)

        mean_1 = mean_u[..., :-2] + batch_u_alpha  # (..., D-2)
        mean_2 = mean_u[..., -1:]  # (..., 1)

        return torch.cat((mean_1, mean_2), dim=-1)  # (..., D-1) i.e. (..., K)

    def mean_theta(self):
        D, = self.D
        size_alpha_k = D - 2

        alpha_k = self.para[..., :-2]  # (..., K-1)
        alpha_beta = self.para[..., -2:]  # (..., 2)

        alpha = self.para[..., -2]  # (..., )
        beta = self.para[..., -1]  # (..., )

        sum_alpha_k = torch.sum(alpha_k, dim=-1)  # (..., )
        sum_alpha_beta = torch.sum(alpha_beta, dim=-1)  # (..., )

        mean_1 = \
        expand_tensor(alpha / (sum_alpha_beta * sum_alpha_k),
                      -1,
                      size_alpha_k)  # (..., K-1)
        mean_1 *= alpha_k

        mean_2 = \
        expand_tensor(beta / sum_alpha_beta,
                      -1,
                      1)  # (..., 1)
        
        return torch.cat((mean_1, mean_2), dim=-1)  # (..., K)
        
    def _bayes_add(self, para, n):
        delta_alpha = torch.sum(n[..., :-1], dim=-1, keepdim=True)  # (..., 1)
        delta = torch.cat((n[..., :-1], delta_alpha, n[..., -1:]), dim=-1)  # (..., K+1)
        return para + delta
    
    def bayes_minus(self, dist):
        delta = self.para - dist.para
        return torch.cat((delta[..., :-2], delta[..., -1:]), dim=-1)
    
    @staticmethod
    def create_para(K: int, val: float = 1.0, batch_grid_dim = tuple()) -> torch.Tensor:
        size = batch_grid_dim + (K+1, )
        para = torch.full(size, val)
        return para


class Generalized_Dirichlet(Multinomial_Conjugator):
    """
    alpha_beta:
      (alpha_1, alpha_2, ..., alpha_K-1)
      (beta_1,  beta_2,  ...., beta_K-1)
    """
    KernelOrder = 2

    def _get_D(self):
        return self.KernelShape

    def _get_K(self):
        _, K = self.KernelShape
        return (K+1, )

    def nat_para(self):
        nat = self.para - 1
        for i in range(self.K[0]-1):
            nat[..., 1, i] -= (self.K[0]-2-i)
        return nat

    def _mean_u(self, para):
        sum_alpha_beta = torch.sum(para, dim=-2, keepdim=True)  # (..., 1, K-1)
        return psi(para) - psi(sum_alpha_beta)

    def log_normalizer(self):
        term_1 = torch.einsum("...ij -> ...", gammaln(self.para))  # (..., )

        sum_alpha_beta = torch.sum(self.para, dim=-2)  # (..., K-1)
        term_2 = torch.sum(gammaln(sum_alpha_beta), dim=-1)  # (..., )

        return term_1 - term_2

    def mean_log_theta(self):
        mean_u = self.mean_u()

        alpha_k = mean_u[..., 0, :]  # (..., K-1)
        beta_k = mean_u[..., 1, :]  # (..., K-1)
        beta_cumsum = torch.cumsum(beta_k, dim=-1)  # (..., K-1)

        mean_log_theta_K = beta_cumsum[..., -1:]  # (..., 1)
        mean_log_theta_k = beta_cumsum[..., :-1]  # (..., K-2)

        leading_zero = torch.zeros(mean_log_theta_K.shape)  # (..., 1)
        mean_log_theta_k = \
        torch.cat((leading_zero, mean_log_theta_k), dim=-1)  # (..., K-1)

        mean_log_theta_k += alpha_k  # (..., K-1)

        return torch.cat((mean_log_theta_k, mean_log_theta_K), dim=-1)  # (..., K)

    def mean_theta(self):
        alpha_k = self.para[..., 0, :]  # (..., K-1)
        beta_k = self.para[..., 1, :]  # (..., K-1)
        sum_a_b = torch.sum(self.para, dim=-2)  # (..., K-1)

        a_by_sum = alpha_k / sum_a_b  # (..., K-1)
        b_by_sum = beta_k / sum_a_b  # (..., K-1)

        b_by_sum_cumprod = torch.cumprod(b_by_sum, dim=-1)  # (..., K-1)
        leading_one = torch.ones(a_by_sum[..., -1:].shape)  # (..., 1)

        term_1 = torch.cat((a_by_sum, leading_one), dim=-1)  # (..., K)
        term_2 = torch.cat((leading_one, b_by_sum_cumprod), dim=-1)  # (..., K)

        return term_1 * term_2

    def _bayes_add(self, para, n):
        n_for_alpha = n[..., :-1]  # (..., K-1)
        n_for_beta = torch.cumsum(n[..., 1:].flip(-1), dim=-1).flip(-1)  # (..., K-1)

        n_for_para = torch.stack([n_for_alpha, n_for_beta], -2)  # (..., 2, K-1)

        return para + n_for_para
    
    def bayes_minus(self, dist):
        delta = self.para - dist.para  # (..., 2, K-1)
        n_1 = delta[..., 0, :]  # (..., K-1)
        n_2 = delta[..., 1, :][..., -1:]  # (..., 1)
        return torch.cat((n_1, n_2), dim=-1)  # (..., K)
    
    @staticmethod
    def create_para(K: int, val: float = 1.0, batch_grid_dim = tuple()) -> torch.Tensor:
        size = batch_grid_dim + (2, K-1)
        para = torch.full(size, val)
        return para
        

def main():
    pass

if __name__ == '__main__':
    main()
    