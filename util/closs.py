import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def cor(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def reduction_loss(embedding, identity_matrix, size, config):
    # Orthogonality
    orthogonal_loss = torch.mean(torch.abs(torch.triu(cor(embedding), diagonal=1))) 
    variance_loss = 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1))) 
    reg_loss = torch.mean(torch.abs(embedding))

    orthogonal_loss = orthogonal_loss * config.orth_weight
    variance_loss = variance_loss * config.var_weight
    reg_loss = reg_loss * config.reg_weight

    loss = orthogonal_loss + variance_loss + reg_loss
    return loss


def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim


class EncodingLoss(nn.Module):
    def __init__(self, dim=64, p =0.8, config = None):
        super(EncodingLoss, self).__init__()
        self.identity_matrix = torch.tensor(np.identity(dim)).float().cuda()
        self.p = p 
        self.dim = dim
        self.config = config # additional configs
        
        self.mmd_loss = MMDLoss(kernel_num = config.mmd_kernel_num)

    def forward(self, atac_embeddings, rna_embeddings):
        # rna
        rna_embedding_cat = []
        rna_reduction_loss = 0
        for i in range(len(rna_embeddings)):                
            rna_embedding_cat.append(rna_embeddings[i])          
            rna_reduction_loss += reduction_loss(rna_embeddings[i], self.identity_matrix, self.dim, self.config)    
        rna_embedding_cat = torch.cat(rna_embedding_cat, 0)        
        rna_reduction_loss /= len(rna_embeddings)
        
        # atac
        atac_reduction_loss = 0
        for i in range(len(atac_embeddings)):
            atac_reduction_loss +=  reduction_loss(atac_embeddings[i], self.identity_matrix, self.dim, self.config)
                                

        atac_reduction_loss /= len(atac_embeddings)
        
        # cosine similarity loss    
        sim_loss = 0   
        for i in range(len(atac_embeddings)):
            top_k_sim = torch.topk(
                torch.max(cosine_sim(atac_embeddings[i], rna_embedding_cat), dim=1)[0],
                int(atac_embeddings[i].shape[0] * self.p))
            sim_loss += torch.mean(top_k_sim[0])
        
        sim_loss = sim_loss/len(atac_embeddings)
        sim_loss *= self.config.sim_weight
        
        # Maximum mean discrepancy
        mmd_loss = 0
        for i in range(len(rna_embeddings)):
            mmd_loss += self.mmd_loss(rna_embeddings[i], atac_embeddings[i])
        mmd_loss *= self.config.mmd_weight    

        loss = rna_reduction_loss + atac_reduction_loss - sim_loss + mmd_loss
        #loss = rna_reduction_loss + atac_reduction_loss + mmd_loss
        #loss = rna_reduction_loss + atac_reduction_loss

        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes=20, feat_dim=64, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, embeddings, labels):
        center_loss = 0
        for i, x in enumerate(embeddings):
            label = labels[i].long()
            batch_size = x.size(0)
            distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
            distmat.addmm_(1, -2, x, self.centers.t())
            distmat = torch.sqrt(distmat)

            classes = torch.arange(self.num_classes).long()
            if self.use_gpu: classes = classes.cuda()
            label = label.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = label.eq(classes.expand(batch_size, self.num_classes))

            dist = distmat * mask.float()
            center_loss += torch.mean(dist.clamp(min=1e-12, max=1e+12))
        
        center_loss = center_loss/len(embeddings)
        return center_loss



class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()

    def forward(self, rna_cell_out, rna_cell_label):
        rna_cell_loss = F.cross_entropy(rna_cell_out, rna_cell_label.long())
        return rna_cell_loss


