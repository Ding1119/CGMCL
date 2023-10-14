import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def contrastive_loss(emb, adj1, adj2, label, emb1, emb2, diag):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    # embeddings1_normalized = F.normalize(emb1, p=2, dim=1)
    # embeddings2_normalized = F.normalize(emb2, p=2, dim=1)
    # similarity_matrix = torch.matmul(embeddings1_normalized, embeddings2_normalized.t())
    similarity_matrix = emb @ emb.T
    # similarity_matrix = emb1 @ emb2.T
    margin = 0.03
    # similarity_matrix = torch.mm(emb1, emb2.t())
    label = F.one_hot(label , num_classes = 3).to(device)
    
    similarity_matrix = similarity_matrix.to(device)
    batch_size = similarity_matrix.size(0)
#     eye = torch.eye(batch_size, device=similarity_matrix.device)
    # import pdb;pdb.set_trace()
    # positive_pairs = similarity_matrix * (1 - eye)  # Remove self-loops
    # negative_pairs = similarity_matrix * eye

   
    # positive_pairs = (similarity_matrix * adj1) + (similarity_matrix * adj2)
    # negative_pairs = (similarity_matrix * (1- adj1)) + (similarity_matrix * (1- adj2))
    adj1 = adj1.to(device)
    adj2 = adj2.to(device)
    positive_pairs = similarity_matrix * (adj1 + adj2) 
    positive_pairs = positive_pairs.to(device)
    negative_pairs = similarity_matrix * ((1- adj1) + (1-adj2)) 
    negative_pairs = negative_pairs.to(device)
    
    
    
    # import pdb;pdb.set_trace()
    # positive_pairs = torch.mm(similarity_matrix, adj1.to(torch.float32))
    # negative_pairs = torch.mm(similarity_matrix, (1-adj2).to(torch.float32))
    
    # positive_loss = torch.sum(torch.mm((1- positive_pairs), label.to(torch.float32)))
    # negative_loss = torch.sum(torch.mm(torch.clamp(negative_pairs - margin, min=0) ** 2, (1 - label.to(torch.float32))) )

    
 
    # positive_loss = torch.sum((1 - positive_pairs) ** 2 )
    # negative_loss = torch.sum(torch.clamp(negative_pairs - margin, min=0) ** 2 )

    

    positive_sum = torch.sum(torch.matmul((positive_pairs), label.to(torch.float32)))
    negative_sum = torch.sum( torch.matmul(torch.clamp(negative_pairs - margin, min=0) ** 2, (1 - label.to(torch.float32)))) 
    
    
    
    positive_loss = -torch.log(positive_sum  * (positive_sum **(-1)) + 1e-8).mean()
    negative_loss = -torch.log(negative_sum  * (negative_sum **(-1)) + 1e-8).mean()
    # similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
    # positive_pairs = torch.sum(similarity_matrix * label, 1)

    # negative_pairs = similarity_matrix * (1- adj)

    # Calculate the contrastive loss
    # loss = torch.sum((1 - positive_pairs) ** 2) + torch.sum(torch.clamp(negative_pairs - margin, min=0) ** 2  + 1e-8)
    # loss = torch.mean((1 - positive_pairs) ** 2) + torch.sum(torch.clamp(negative_pairs - margin, min=0) ** 2)
    # import pdb;pdb.set_trace()
    similarity_matrix = torch.exp(similarity_matrix / margin)
    similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
    # penalty_term = torch.sum(similarity_matrix)
    penalty_term = torch.mean(torch.sum(torch.exp(positive_pairs) / torch.sum(torch.exp(negative_pairs), dim=1, keepdim=True), dim=1))
    penalty_coeff = 0.001
    # loss = positive_loss + negative_loss 
    diagonal_loss = mse_loss(torch.matmul(emb1, emb2.t()), diag)
    similarity_scores = torch.matmul(emb1, emb2.t())  # (batch_size, batch_size)
    similarity_probs = torch.sigmoid(similarity_scores)
                    
    adj = adj1 + adj2
                            # 計算相似性損失
    similarity_loss = bce_loss(similarity_scores, adj)
    # loss = positive_loss  + negative_loss + similarity_loss + diagonal_loss
    loss1 = -torch.log(positive_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
    loss2 = -torch.log(negative_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
    loss = loss1 + loss2 + diagonal_loss
    # loss = positive_loss + negative_loss 
    # loss = torch.log(loss + 1e-8)
    return loss / ((2 * batch_size) **2)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight.to(device)
        self.reduction = reduction

    def forward(self, inputs, targets):
#         import pdb;pdb.set_trace()
        
        loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)
        return loss
    
def info_loss(emb, adj1, adj2, label):
    
    similarity_matrix = emb @ emb.T
    # similarity_matrix = emb1 @ emb2.T
    margin = 0.03

    num_classes = 3
    label = F.one_hot(label , num_classes = 3)
    

    batch_size = similarity_matrix.size(0)
    eye = torch.eye(batch_size, device=similarity_matrix.device)
   
    batch_size = similarity_matrix.size(0)
    similarity_matrix = similarity_matrix.to(device)
    positive_pairs = similarity_matrix * (adj1 + adj2) 
    negative_pairs = similarity_matrix * ((1- adj1) + (1-adj2))  
    
    # penalty_term = torch.sum(similarity_matrix)
    # penalty_term = torch.mean(torch.sum(torch.exp(positive_pairs) / torch.sum(torch.exp(negative_pairs), dim=1, keepdim=True), dim=1))
    # penalty_coeff = 0.001
    
    similarity_matrix_sum = torch.sum(positive_pairs, 1, keepdim=True)
    positives_sum = torch.matmul(similarity_matrix,label.to(torch.float32))
    loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()

    
    return loss / batch_size


class MGECLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(MGECLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, view1_embeddings, view2_embeddings, adjacency_matrix, diagonal_matrix):
        # 計算相似性概率
        
        similarity_scores = torch.matmul(view1_embeddings, view2_embeddings.t())  # (batch_size, batch_size)
        similarity_probs = torch.sigmoid(similarity_scores)
        batch_size = similarity_scores.size(0)
        
        # 計算相似性損失
        similarity_loss = self.bce_loss(similarity_scores, adjacency_matrix)
        
        # 計算嵌入一致性損失
        embedding_consistency_loss = self.mse_loss(view1_embeddings, view2_embeddings)
        
        # 計算分塊對角表示損失
        diagonal_loss = self.mse_loss(torch.matmul(view1_embeddings, view1_embeddings.t()), diagonal_matrix)
        
        # 總損失
        total_loss = self.alpha * similarity_loss + self.beta * embedding_consistency_loss + self.gamma * diagonal_loss
        
        return total_loss /batch_size

    
class SACLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SACLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, adjacency_matrix):
        # 計算相似性概率
        similarity_matrix = torch.matmul(embeddings, embeddings.t())  # (batch_size, batch_size)
        similarity_matrix = similarity_matrix.to(device)
        similarity_matrix /= self.temperature
        similarity_probs = F.softmax(similarity_matrix, dim=1)
        batch_size = similarity_matrix.size(0)

        # 計算正樣本概率
        pos_probs = similarity_probs * adjacency_matrix
        pos_probs = pos_probs.sum(dim=1)

        # 計算負樣本概率
        neg_probs = similarity_probs * (1 - adjacency_matrix)
        neg_probs = neg_probs.sum(dim=1)

        # 計算對比損失
        loss = -torch.log(pos_probs / (pos_probs + neg_probs)).mean()

        return loss /batch_size 
