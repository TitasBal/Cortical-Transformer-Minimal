import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    """Configuration parameters for the GPT model.

    Attributes:
        block_size: Maximum sequence length (context size).
        vocab_size: Number of tokens in the vocabulary.
        n_layer: Number of transformer layers (blocks).
        n_head: Number of attention heads.
        n_embd: Embedding dimension for tokens and positions.
        dropout: Dropout probability for regularization.
        tau_att: Fixed time constant for attention dynamics (alpha).
        tau_v: Fixed time constant for value dynamics (nu).
        dt: Time step for ODE simulation (Euler integration).
        T: Number of simulation steps per forward pass in dynamic mode.
    """
    block_size: int = 64
    vocab_size: int = 10
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    # --- Dynamic Params ---
    tau_att: float = 2.0
    tau_v: float = 2.0
    dt: float = 0.1
    T: int = 5

class DynamicCausalSelfAttention(nn.Module):
    """Dynamic Causal Self-Attention layer.

    Attributes:
        c_attn: Linear layer for combined query, key, value projections.
        c_proj: Linear layer for output projection.
        attn_dropout: Dropout layer for attention weights.
        resid_dropout: Dropout layer for the output projection.
        tau_alpha: Time constant for attention dynamics.
        tau_nu: Time constant for value dynamics.
        bias: Lower-triangular mask buffer for causal attention.
    """
    def __init__(self, config: GPTConfig):
        """Initializes the DynamicCausalSelfAttention layer.

        Args:
            config: An instance of GPTConfig containing model hyperparameters.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by num heads"
        self.config = config
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        assert config.tau_att > 0 and config.tau_v > 0, "Time constants must be positive"
        self.tau_alpha = config.tau_att
        self.tau_nu = config.tau_v

        # Causal mask buffer
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def get_initial_states(self, B: int, T_ctx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates zero-initialized states for dynamic simulation.

        Args:
            B: Batch size.
            T_ctx: Context length (sequence length).
            device: The torch device to create tensors on.

        Returns:
            A tuple containing:
                - alpha_state: Initial attention state tensor (zeros).
                - nu_state: Initial value state tensor (zeros).
        """
        # Initial attention (alpha) and value (nu) states
        alpha = torch.zeros(B, self.n_head, T_ctx, T_ctx, device=device, dtype=torch.float32)
        nu = torch.zeros(B, self.n_head, T_ctx, self.head_size, device=device, dtype=torch.float32)
        return alpha, nu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the standard static causal self-attention forward pass.

        Args:
            x: Input tensor of shape (B, T_ctx, C).

        Returns:
            Output tensor of shape (B, T_ctx, C).
        """
        B, T_ctx, C = x.size()
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        k = k.view(B, T_ctx, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_ctx, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_ctx, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        # Attention calculation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T_ctx,:T_ctx] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att) # Prevent NaNs
        att = self.attn_dropout(att)

        # Output calculation
        y = (att @ v).transpose(1, 2).contiguous().view(B, T_ctx, C) # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def step(self, x: torch.Tensor, current_alpha: torch.Tensor, current_nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one step of the dynamic ODE simulation using Euler integration.

        Args:
            x: Input tensor of shape (B, T_ctx, C).
            current_alpha: Current attention state tensor (B, nh, T_ctx, T_ctx).
            current_nu: Current value state tensor (B, nh, T_ctx, hs).

        Returns:
            A tuple containing:
                - y: Output tensor for this step (B, T_ctx, C).
                - next_alpha: Updated attention state tensor.
                - next_nu: Updated value state tensor.
        """
        B, T_ctx, C = x.size()
        dt = self.config.dt
        tau_alpha, tau_nu = self.tau_alpha, self.tau_nu

        # Calculate target values (instantaneous Q, K, V, attention)
        q, k, nu_target = self.c_attn(x).split(self.config.n_embd, dim=2)
        k = k.view(B, T_ctx, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T_ctx, self.n_head, self.head_size).transpose(1, 2)
        nu_target = nu_target.view(B, T_ctx, self.n_head, self.head_size).transpose(1, 2)

        att_raw_target = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_raw_target = att_raw_target.masked_fill(self.bias[:, :, :T_ctx, :T_ctx] == 0, float('-inf'))
        alpha_target = F.softmax(att_raw_target, dim=-1)
        alpha_target = torch.nan_to_num(alpha_target)


        delta_nu = (1. / tau_nu) * (-current_nu + nu_target)
        nu_breve = torch.nan_to_num(current_nu + tau_nu * delta_nu)
        next_nu = current_nu + dt * delta_nu

        delta_alpha = (1. / tau_alpha) * (-current_alpha + alpha_target)
        alpha_breve = torch.nan_to_num(current_alpha + tau_alpha * delta_alpha)
        next_alpha = current_alpha + dt * delta_alpha


        alpha_breve_dropped = self.attn_dropout(alpha_breve)

        # Output: prospective_attention @ prospective_value
        y = (alpha_breve_dropped @ nu_breve).transpose(1, 2).contiguous().view(B, T_ctx, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, next_alpha, next_nu

class MLP(nn.Module):
    """Simple Feed-Forward Network (MLP) block used in Transformer layers."""
    def __init__(self, config: GPTConfig):
        """Initializes the MLP layers.

        Args:
            config: An instance of GPTConfig containing model hyperparameters.
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for the MLP."""
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Step function for MLP (identical to forward as it's stateless)."""
        return self.forward(x)

class Block(nn.Module):
    """A single Transformer Block.

    Combines Dynamic Causal Self-Attention and an MLP with residual connections
    and layer normalization. Supports both static (`forward`) and dynamic (`step`) modes.
    """
    def __init__(self, config: GPTConfig):
        """Initializes the Transformer Block.

        Args:
            config: An instance of GPTConfig containing model hyperparameters.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = DynamicCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def get_initial_states(self, B: int, T_ctx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets initial dynamic states from the attention sub-module.

        Args:
            B: Batch size.
            T_ctx: Context length (sequence length).
            device: The torch device to create tensors on.

        Returns:
            A tuple containing initial alpha and nu states.
        """
        return self.attn.get_initial_states(B, T_ctx, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the standard static forward pass through the block."""
        x = x + self.attn(self.ln_1(x)) # Residual connection + Attention
        x = x + self.mlp(self.ln_2(x))  # Residual connection + MLP
        return x

    def step(self, x: torch.Tensor, alpha_state: torch.Tensor, nu_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one dynamic step through the block using the `step` methods
           of the attention and MLP sub-modules.

        Args:
            x: Input tensor (B, T_ctx, C).
            alpha_state: Current attention state tensor for this block.
            nu_state: Current value state tensor for this block.

        Returns:
            A tuple containing:
                - x_out: Output tensor after the dynamic step (B, T_ctx, C).
                - next_alpha: Updated attention state.
                - next_nu: Updated value state.
        """
        attn_out, next_alpha, next_nu = self.attn.step(self.ln_1(x), alpha_state, nu_state)
        x = x + attn_out # Apply residual connection for attention
        x = x + self.mlp.step(self.ln_2(x)) # Apply residual connection for MLP
        return x, next_alpha, next_nu

class GPT(nn.Module):
    """The main Generative Pre-trained Transformer (GPT) model class.

    Integrates embeddings, multiple Transformer Blocks, and a final prediction head.
    Can operate in standard static mode (`forward`) or dynamic simulation mode (`step`).
    Can be configured for Language Modeling or Sequence Classification.
    """

    def __init__(self, config: GPTConfig, num_classes: Optional[int] = None):
        """Initializes the GPT model.

        Args:
            config: An instance of GPTConfig containing model hyperparameters.
            num_classes: If specified (e.g., 10 for ListOps), configures the model
                         for classification with that many output classes. Otherwise (None),
                         configures for language modeling.
        """
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        # --- Store num_classes ---
        self.num_classes = num_classes

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # drop = nn.Dropout(config.dropout), # Consider adding dropout after embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # --- Conditional Head Creation ---
        if self.num_classes is not None:
            # --- Classification Mode ---
            print(f"Model configured for Classification with {self.num_classes} classes.")
            self.lm_head = None # Not used for classification
            # Create a linear layer mapping final embedding to class scores
            self.classification_head = nn.Linear(config.n_embd, self.num_classes)
            # Initialize weights for the new head
            self._init_weights(self.classification_head)
        else:
            # --- Language Modeling Mode ---
            print("Model configured for Language Modeling.")
            # Standard LM head mapping to vocabulary size
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.classification_head = None
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights) # Apply weight initialization

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model initialized: {n_params/1e6:.2f}M parameters")
        print(f"Dynamic config: T={config.T}, dt={config.dt}, tau_att={config.tau_att:.2f}, tau_v={config.tau_v:.2f}, Prospective={getattr(config, 'use_prospective_coding', True)}") # Added prospective check

    def _init_weights(self, module):
        """Initializes weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
             # Check if the layer exists before initializing
            if module is not None and hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module is not None and hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
             if module is not None and hasattr(module, 'weight'):
                nn.init.ones_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def _prepare_input(self, idx: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, T_seq = idx.size()
        if T_seq > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T_seq = self.config.block_size
        pos = torch.arange(0, T_seq, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        return x, T_seq

    def _get_logits_and_loss(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calculates final logits and loss based on task type."""
        if self.num_classes is not None: # --- Classification Task ---
            assert self.classification_head is not None, "num_classes was set but classification_head is None"
            # Use embedding of the last token for classification prediction
            class_token_embedding = x[:, -1, :] # Shape: (B, C)
            logits = self.classification_head(class_token_embedding) # Shape: (B, num_classes)
            loss = None
            if targets is not None:
                # Targets should have shape (B,) for CrossEntropyLoss with logits (B, num_classes)
                loss = F.cross_entropy(logits.view(-1, self.num_classes), targets.view(-1))
        else: # --- Language Modeling Task ---
            assert self.lm_head is not None, "num_classes is None but lm_head is None"
            loss = None
            if targets is not None: # Training/Validation
                logits = self.lm_head(x) # Shape: (B, T_ctx, V)
                # Align targets if input was cropped by _prepare_input
                if targets.size(1) != logits.size(1):
                     targets = targets[:, -logits.size(1):]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else: # Inference (like generate)
                logits = self.lm_head(x[:, [-1], :]) # Shape: (B, 1, V)
        return logits, loss

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, _ = self._prepare_input(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self._get_logits_and_loss(x, targets)

    def step(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, t_seq = self._prepare_input(idx)
        B, device = x.size(0), x.device
        alpha_states, nu_states = [], []
        for block in self.transformer.h:
            alpha, nu = block.get_initial_states(B, t_seq, device)
            alpha_states.append(alpha); nu_states.append(nu)

        for _ in range(self.config.T):
            next_alpha_states, next_nu_states = [], []
            current_x = x
            for i, block in enumerate(self.transformer.h):
                current_x, next_alpha, next_nu = block.step(current_x, alpha_states[i], nu_states[i])
                next_alpha_states.append(next_alpha); next_nu_states.append(next_nu)
            alpha_states = next_alpha_states
            nu_states = next_nu_states
            x = current_x

        x = self.transformer.ln_f(x)
        return self._get_logits_and_loss(x, targets)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, dynamic: bool = False) -> torch.Tensor:
        self.eval()
        forward_method = self.step if dynamic else self.forward
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = forward_method(idx_cond, targets=None)
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

    # configure_optimizers method remains largely the same, ensure it handles None heads
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Filter out None parameters just in case
        param_dict = {pn: p for pn, p in param_dict.items() if p is not None}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer