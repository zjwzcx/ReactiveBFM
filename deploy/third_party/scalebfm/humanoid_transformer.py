import math
import torch
import torch.nn as nn

class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1000, base=10000):
        super(RoPEPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # Create position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Calculate frequencies
        freqs = torch.outer(position, self.inv_freq)  # (seq_len, dim//2)
        
        # Create rotation matrices
        cos_freqs = torch.cos(freqs)  # (seq_len, dim//2)
        sin_freqs = torch.sin(freqs)  # (seq_len, dim//2)
        
        # Apply RoPE rotation
        x_rotated = self.apply_rope(x, cos_freqs, sin_freqs)

        return x_rotated
    
    def apply_rope(self, x, cos_freqs, sin_freqs):
        # x shape: (batch_size, seq_len, dim)
        # cos_freqs, sin_freqs shape: (seq_len, dim//2)
        
        batch_size, seq_len, dim = x.shape
        
        # Reshape x to separate even and odd dimensions
        x_even = x[:, :, 0::2]  # (batch_size, seq_len, dim//2)
        x_odd = x[:, :, 1::2]   # (batch_size, seq_len, dim//2)
        
        # Expand cos_freqs and sin_freqs to match batch dimension
        cos_freqs = cos_freqs.unsqueeze(0)  # (1, seq_len, dim//2)
        sin_freqs = sin_freqs.unsqueeze(0)  # (1, seq_len, dim//2)
        
        # Apply rotation
        x_even_rotated = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rotated = x_even * sin_freqs + x_odd * cos_freqs
        
        # Interleave the rotated dimensions
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (..., dim)
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU activation function: SwiGLU(x) = SiLU(xW) ⊗ (xV)
    where SiLU(x) = x * sigmoid(x) (also known as Swish)
    """
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        # x shape: (..., input_dim)
        # SiLU activation: x * sigmoid(x)
        swish_gate = self.silu(self.w(x))
        # Element-wise multiplication with the value branch
        gated = swish_gate * self.v(x)
        # Project back to input dimension
        return self.output(gated)

class HumanoidTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.feed_forward = SwiGLU(embed_dim, ff_dim)

        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.rmsnorm3 = RMSNorm(embed_dim)
        self.cond_norm = RMSNorm(embed_dim)
        
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(self, x, c, self_attn_mask = None):
        
        x_norm = self.rmsnorm1(x)
        x_rope = self.rope(x_norm)
        attn_output, _ = self.self_attention(x_rope, x_rope, x_norm, attn_mask=self_attn_mask)
        x = x + attn_output

        x_norm2 = self.rmsnorm2(x)
        c_norm = self.cond_norm(c)
        cross_output, _ = self.cross_attention(
            query=x_norm2,
            key=c_norm,
            value=c_norm
        )
        x = x + cross_output

        x_norm3 = self.rmsnorm3(x)
        ff_output = self.feed_forward(x_norm3)
        x = x + ff_output

        return x

class TaskEmbedder(nn.Module):
    def __init__(self,
                 task_obs_dim,
                 embedding_dim,
                 reduced_task_dim = None,
                 hidden_dims = None):
        super().__init__()

        if reduced_task_dim is not None:
            self.task_projection = self._build_task_projection(task_obs_dim, reduced_task_dim, hidden_dims)
            A = torch.randn(embedding_dim, reduced_task_dim, dtype=torch.float)
            Q, R = torch.linalg.qr(A, mode="reduced")
            diag = torch.sign(torch.diag(R))
            diag[diag==0] = 1.0
            self.register_buffer("W", Q * diag)
            self._forward_method = self._reduced_task_projection
        else:
            self.task_projection = self._build_task_projection(task_obs_dim, embedding_dim, hidden_dims)
            self._forward_method = self._normal_task_projection

    def _build_task_projection(self, input_dim, output_dim, hidden_dims = None):
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Linear(input_dim, output_dim)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ELU())

        for layer_index in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index+1]))
            layers.append(nn.ELU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def _reduced_task_projection(self, task_obs):
        task_embedding = self.task_projection(task_obs)
        task_embedding = task_embedding / (task_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        task_tokens = torch.matmul(task_embedding, self.W.T)
        return task_tokens
    
    def _normal_task_projection(self, task_obs):
        task_tokens = self.task_projection(task_obs)
        return task_tokens
    
    def forward(self, task_obs):
        return self._forward_method(task_obs)
    
    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                in_dim = m.weight.size(1)
                std = 1.0 / math.sqrt(in_dim)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class HumanoidTransformer(nn.Module):
    def __init__(self, 
                 prop_obs_dim, 
                 action_dim, 
                 output_dim,
                 embed_dim = 256, 
                 num_heads = 4, 
                 ff_dim = 256, 
                 num_layers = 4,
        ):
        super(HumanoidTransformer, self).__init__()
        self.prop_obs_dim = prop_obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.prop_projection = nn.Linear(prop_obs_dim, embed_dim)
        self.action_projection = nn.Linear(action_dim, embed_dim)
        
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            HumanoidTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(embed_dim)
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(self, prop_obs, action_obs, task_tokens):
        batch_size = prop_obs.shape[0]
        context_size = prop_obs.shape[1]

        prop_obs = self.prop_projection(prop_obs) # (bs, cl, dim)
        action_obs = self.action_projection(action_obs) # (bs, cl, dim)

        context = prop_obs.new_empty(batch_size, 2*context_size-1, self.embed_dim)
        context[:, 0::2] = prop_obs
        context[:, 1::2] = action_obs[:, 1:]

        empty_embedding = self.empty_embedding.expand(batch_size, -1, -1)
        x = torch.cat([context, empty_embedding], dim=1)
        
        self_attn_mask = torch.zeros(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        # self_attn_mask[:-1, -1] = True # weishuai: tensorrt does not support dynamic index
        row_idx = torch.arange(x.shape[1] - 1, device=x.device)
        col_idx = torch.full((x.shape[1] - 1,), x.shape[1] - 1, device=x.device)
        self_attn_mask[row_idx, col_idx] = True
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, task_tokens, self_attn_mask=self_attn_mask)
        x = self.final_norm(x)

        empty_embedding = x[:, -1, :]
        output = self.projection_head(empty_embedding)
        
        return output

    @torch.no_grad()
    def init_weights(self, head_init_val = None):
        L = len(self.transformer_blocks)
        res_scale = 1.0 / math.sqrt(2.0 * L)
        
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                continue
            
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.scale)
            elif isinstance(m, nn.Linear):
                in_dim = m.weight.size(1)
                std = 1.0 / math.sqrt(in_dim)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                embed_dim = m.embed_dim
                std = 1.0 / math.sqrt(embed_dim)

                nn.init.normal_(m.in_proj_weight, mean=0.0, std=std)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                
                out_std = std / math.sqrt(2 * len(self.transformer_blocks))
                nn.init.normal_(m.out_proj.weight, mean=0.0, std=out_std)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        for blk in self.transformer_blocks:
            if hasattr(blk, "feed_forward") and hasattr(blk.feed_forward, "output"):
                blk.feed_forward.output.weight.mul_(res_scale)

        nn.init.trunc_normal_(self.empty_embedding, std=0.02)

        if head_init_val is not None:
            nn.init.zeros_(self.projection_head.weight)
            self.projection_head.bias.copy_(head_init_val)
        else:
            nn.init.zeros_(self.projection_head.weight)
            nn.init.zeros_(self.projection_head.bias)