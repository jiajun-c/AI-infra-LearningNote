import torch
import torch.nn as nn
import torch.nn.functional as F
import math

N, d = 16, 8

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

Br,Bc = 4,d

O = torch.zeros((N, d))
l = torch.zeros((N, 1))
m = torch.full((N, 1), -torch.inf)

for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    Kj = K_mat[block_start_Bc:block_end_Bc, :]
    Vj = V_mat[block_start_Bc:block_end_Bc, :]

    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        mi = m[block_start_Br:block_end_Br, :]
        li = l[block_start_Br:block_end_Br, :]
        Oi = O[block_start_Br:block_end_Br, :]
        Qi = Q_mat[block_start_Br:block_end_Br, :]
        
        Sij = Qi @ Kj.T

        mij_hat = torch.max(Sij, dim=1).values[:, None]

        pij_hat = torch.exp(Sij - mij_hat)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
        li_new = torch.exp(mi-mi_new) * li * torch.exp(mij_hat - mi_new)*lij_hat
        Oi = (li * torch.exp(mi - mi_new)*Oi/li_new) * (torch.exp(mij_hat - mi_new)*pij_hat/li_new) @ Vj

        m[block_start_Br:block_end_Br, :] = mi_new
        l[block_start_Br:block_end_Br, :] = li_new

        O[block_start_Br:block_end_Br, :] = Oi



