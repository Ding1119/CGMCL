import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight.to(device)
        self.reduction = reduction

    def forward(self, inputs, targets):

        loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)
        return loss


def contrastive_loss(emb, adj1, adj2, label, emb1, emb2, diag, n_classes, margin):
    margin = 0.03
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    similarity_matrix = emb @ emb.T
    label = F.one_hot(label, n_classes).to(device)
    similarity_matrix = similarity_matrix.to(device)
    batch_size = similarity_matrix.size(0)

    adj1 = adj1.to(device)
    adj2 = adj2.to(device)

    adj_sum = adj1 + adj2
    threshold_value = 0.5 
    thresholded_adj = (adj_sum >= threshold_value).float()

    positive_pairs = similarity_matrix * thresholded_adj
    positive_pairs = positive_pairs.to(device)

    adj_sum_neg = (1-adj1) + (1- adj2)
    thresholded_adj_neg = (adj_sum_neg >= threshold_value).float()

    negative_pairs = similarity_matrix * thresholded_adj_neg
    negative_pairs = negative_pairs.to(device)
    

    positive_sum = torch.sum(torch.matmul(positive_pairs, label.float()))
    negative_sum = torch.sum(torch.matmul(torch.clamp(negative_pairs - margin, min=0) ** 2, (1 - label.float())))

    
    similarity_matrix = torch.exp(similarity_matrix / margin)

    similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)

    diagonal_loss = mse_loss(torch.matmul(emb1, emb2.t()), diag)

    loss1 = -torch.log(positive_sum * (similarity_matrix_sum ** (-1)) + 1e-8).mean()
    loss2 = -torch.log(negative_sum * (similarity_matrix_sum ** (-1)) + 1e-8).mean()
    loss = loss1 + loss2 + diagonal_loss


    return loss / ((2 * batch_size) ** 2)