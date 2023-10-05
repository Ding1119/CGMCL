import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from sklearn.cluster import KMeans
from keras.applications.mobilenet import MobileNet, preprocess_input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1024)
        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
class GraphConvolution_img(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution_img, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        # self.linear2 = nn.Linear( 65536, 12)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, adjacency):
        # x = x.T @ self.linear(x)
       
        x = self.linear2(x)
        # import pdb;pdb.set_trace()
        x = torch.matmul(adjacency, x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)

        return x
    
class GraphConvolution_f(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution_f, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        # self.linear2 = nn.Linear( 65536, 12)
        self.linear2 = nn.Linear(256, 12)
        self.linear3 = nn.Linear(12, 3)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, adjacency):
        # x = x.T @ self.linear(x)
        
        # x = self.linear2(x)
        # import pdb;pdb.set_trace()
        x = torch.matmul(adjacency, x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        # import pdb;pdb.set_trace()
        return x
    
class AttentionNetwork_img(nn.Module):
    def __init__(self, num_features):
        super(AttentionNetwork_img, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        # x = F.dropout(x, p=0.5)
        x = self.sigmoid(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        attention_weights = self.sigmoid(x)
        return attention_weights
    
class AttentionNetwork_f(nn.Module):
    def __init__(self, num_features):
        super(AttentionNetwork_f, self).__init__()
        self.fc1 = nn.Linear(num_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        # x = F.dropout(x, p=0.5)
        x = self.sigmoid(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        attention_weights = self.sigmoid(x)
        return attention_weights


class Projection(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(Projection, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = nn.ReLU()
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, X, adj):
        h = self.W(X)
        N = h.size(0)
        
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = F.relu(self.a(a_input).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        
        h_prime = torch.matmul(attention, h)
        return h_prime

# 定義 Graph Attention Network
class GAT_img(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GAT_img, self).__init__()
        self.attention1 = GraphAttentionLayer(in_features, hidden_features)
        self.attention2 = GraphAttentionLayer(hidden_features, num_classes)

    def forward(self, X, adj):
        h = self.attention1(X, adj)
        h = F.relu(h)
        logits = self.attention2(h, adj)
        return logits
    
class GAT_f(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GAT_f, self).__init__()
        self.attention1 = GraphAttentionLayer(in_features, hidden_features)
        self.attention2 = GraphAttentionLayer(hidden_features, num_classes)

    def forward(self, X, adj):
        # h =  torch.matmul(adj, X)
        
        h = self.attention1(X, adj)
        h = F.relu(h)
        logits = self.attention2(h, adj)
        return logits
    
class Model_PD(nn.Module):
    def __init__(self, projection, input_resnet, n_classes):
        super(Model_PD, self).__init__()
        #self.encoder = CNNEncoder()
        self.cnn_encoder = CNN()
        self.input_resnet = input_resnet
        self.n_classes = n_classes
        self.kmeans = KMeans(n_clusters=3) 
        # self.cnn_encoder = DeepEncoder() #torch.Size([300, 3, 64, 64])
        # self.cnn_encoder = CNNEncoder()
        # self.vgg_encoder = nn.Sequential(*list(vgg16.features.children()))
        self.gcn_img = GraphConvolution_img(64, 3)
        self.gcn_f = GraphConvolution_f(12, 3)
        self.gat_img = GAT_img(1024,256,3)
        self.gat_f = GAT_f(12,5,3)
        self.projection = projection
   
        # self.linear_vgg1 = nn.Linear(65536, 1024)
        # self.linear_vgg2 = nn.Linear(1024, 256)
        self.linear1 = nn.Linear(259, 2)
        self.linear2 = nn.Linear(256, 2)
        self.linear3 = nn.Linear(1030, 256)
        self.linear4 = nn.Linear(256,self.n_classes)
        self.linear_f = nn.Linear(18,self.n_classes)
        self.attention_f = AttentionNetwork_f(15)
        self.attention_img = AttentionNetwork_img(1027)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_f,adjacency_img, adjacency_f):
        # import pdb;pdb.set_trace()
        # x = self.encoder(x) #torch.Size([300, 65536])
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x_encoder = self.cnn_encoder(x)
        
        #x_encoder = self.cnn_encoder(x)
        
        x_gat_img = self.gat_img(x_encoder, adjacency_img)
        # x_encoder = self.vgg_encoder(x)
        # x_encoder = torch.flatten(x_encoder, start_dim=1)
        # x_encoder = self.linear_vgg1(x_encoder)
        # x_encoder = self.linear_vgg2(x_encoder)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x_gat_f = self.gat_f(x_f, adjacency_img)
        # import pdb;pdb.set_trace()
        x_gcn_f = self.gcn_f(x_f, adjacency_f)
        # import pdb;pdb.set_trace()
        # kmeans_output_f = self.kmeans.fit_transform(x_gcn_f.detach().numpy())
        # kmeans_tensor_f = torch.from_numpy(kmeans_output_f).float().to(x_gcn_f.device)
        
        x_gcn_img = self.gcn_img(x_encoder, adjacency_img)
        # kmeans_output_img = self.kmeans.fit_transform(x_gcn_img.detach().numpy())
        # kmeans_tensor_img = torch.from_numpy(kmeans_output_img).float().to(x_gcn_img.device)

        x_f_fusion = torch.cat((x_f, x_gat_f), 1)
        x_img = torch.cat((x_encoder, x_gat_img), 1)
       
        
        w_att_f = self.attention_f(x_f_fusion)
        w_att_img = self.attention_img(x_img)
        
        attended_gat_f = x_gat_f * w_att_f
        attended_gat_img = x_gat_img * w_att_img
        x_f_att = torch.cat((x_f_fusion, attended_gat_f), 1) # torch.Size([300, 17])
        x_img_att = torch.cat((x_img, attended_gat_img), 1) # torch.Size([300, 1029])
        
 
        emb1  = self.linear_f(x_f_att)
        # emb1  = self.linear4(emb1)
        
        emb2  = self.linear3(x_img_att)
        emb2  = self.linear4(emb2)
        #import pdb;pdb.set_trace()
        # x1 = self.linear1(x_f)
      

        # # x2 = self.linear2(x_encoder)
        # x2 = self.linear1(x_img)

        # x1 = self.softmax(x1)
        # x2 = self.softmax(x2)

        # x1 = self.linear1(attended_gcn)
        z = emb1 + emb2
        
        #import pdb;pdb.set_trace()
       
        #emb2 =  torch.from_numpy(emb2).to(torch.float32).to(device)
        #import pdb;pdb.set_trace()

    
        #z = torch.from_numpy(z).to(device)
        #emb2 = torch.from_numpy(emb2).float().to(device)
        
    

        #x1 = self.softmax(emb1) 
        #x2 = self.softmax(emb2) 
        
        #import pdb;pdb.set_trace()
    
        z  = self.softmax(z).to(torch.float32)
        
        #import pdb;pdb.set_trace()
        

        return emb1, emb2, z
    
    def nonlinear_transformation(self, h):
        z = self.projection(h)
        return z
