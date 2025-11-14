import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor


class GATLayer(nn.Module):
    """Graph Attention Network Layer
    
    Implements multi-head graph attention mechanism for grid-based graphs.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        h: FloatTensor,
        adj_mask: LongTensor,
    ) -> FloatTensor:
        """
        Parameters
        ----------
        h : FloatTensor
            Node features [b, n_nodes, in_features]
        adj_mask : LongTensor
            Adjacency mask [b, n_nodes, n_nodes], 1 for connected, 0 for not connected
        
        Returns
        -------
        FloatTensor
            Output features [b, n_nodes, out_features]
        """
        b, n, _ = h.shape
        
        # Linear transformation
        Wh = self.W(h)  # [b, n, out_features]
        Wh = Wh.view(b, n, self.num_heads, self.head_dim)  # [b, n, num_heads, head_dim]
        
        # Compute attention scores
        # Wh1 and Wh2 for each head
        Wh1 = Wh.transpose(1, 2)  # [b, num_heads, n, head_dim]
        Wh2 = Wh.transpose(1, 2)  # [b, num_heads, n, head_dim]
        
        # Compute attention coefficients
        e = self._compute_attention_scores(Wh1, Wh2)  # [b, num_heads, n, n]
        
        # Apply mask and LeakyReLU
        adj_mask_expanded = adj_mask.unsqueeze(1)  # [b, 1, n, n]
        e = e.masked_fill(adj_mask_expanded == 0, float('-inf'))
        e = self.leaky_relu(e)
        
        # Softmax
        attention = F.softmax(e, dim=-1)  # [b, num_heads, n, n]
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh1)  # [b, num_heads, n, head_dim]
        h_prime = h_prime.transpose(1, 2).contiguous()  # [b, n, num_heads, head_dim]
        h_prime = h_prime.view(b, n, self.out_features)  # [b, n, out_features]
        
        return h_prime
    
    def _compute_attention_scores(self, Wh1: FloatTensor, Wh2: FloatTensor) -> FloatTensor:
        """Compute attention scores using concatenation method"""
        b, num_heads, n, head_dim = Wh1.shape
        
        # Expand dimensions for broadcasting
        Wh1_expanded = Wh1.unsqueeze(3)  # [b, num_heads, n, 1, head_dim]
        Wh2_expanded = Wh2.unsqueeze(2)  # [b, num_heads, 1, n, head_dim]
        
        # Concatenate
        Wh_concat = torch.cat([Wh1_expanded.repeat(1, 1, 1, n, 1), 
                               Wh2_expanded.repeat(1, 1, n, 1, 1)], dim=-1)  # [b, num_heads, n, n, 2*head_dim]
        
        # Compute attention
        e = torch.matmul(Wh_concat, self.a)  # [b, num_heads, n, n, 1]
        e = e.squeeze(-1)  # [b, num_heads, n, n]
        
        return e


class GAT(nn.Module):
    """Multi-layer Graph Attention Network"""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        
        layers = []
        # First layer
        layers.append(
            GATLayer(in_features, hidden_features, num_heads, dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                GATLayer(hidden_features, hidden_features, num_heads, dropout)
            )
        
        # Last layer
        if num_layers > 1:
            layers.append(
                GATLayer(hidden_features, out_features, num_heads, dropout)
            )
        else:
            # If only one layer, adjust first layer output
            layers[0] = GATLayer(in_features, out_features, num_heads, dropout)
        
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        h: FloatTensor,
        adj_mask: LongTensor,
    ) -> FloatTensor:
        """
        Parameters
        ----------
        h : FloatTensor
            Node features [b, n_nodes, in_features]
        adj_mask : LongTensor
            Adjacency mask [b, n_nodes, n_nodes]
        
        Returns
        -------
        FloatTensor
            Output features [b, n_nodes, out_features]
        """
        for i, layer in enumerate(self.layers):
            h = layer(h, adj_mask)
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
        return h

