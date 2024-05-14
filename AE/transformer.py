import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Linear(pf_dim, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        return src

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([TransformerBlock(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return self.fc_out(src)

# Instantiate the Transformer
model = Transformer(input_dim=512, output_dim=512, hid_dim=2048, n_layers=6, n_heads=8, pf_dim=2048, dropout=0.1, device=torch.device('cpu'))

# Create a dummy input tensor (batch_size=64, src_len=100)
src = torch.randint(0, 512, (64, 100))

# Create a dummy src_mask tensor (batch_size=64, 1, 1, src_len=100)
src_mask = torch.ones(100,64)

# Call the forward method
output = model.forward(src, src_mask)

# Print the output
print(output)