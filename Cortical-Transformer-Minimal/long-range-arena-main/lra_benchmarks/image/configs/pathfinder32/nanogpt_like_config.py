# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration and hyperparameter sweeps for "nanoGPT-like" Transformer on Pathfinder."""

import ml_collections
from lra_benchmarks.image.configs.pathfinder32 import base_pathfinder32_config # Important import

def get_config():
    """Get the hyperparameter configuration."""
    config = base_pathfinder32_config.get_config() # Start with base Pathfinder settings

    # Override with nanoGPT-like Transformer settings
    config.model_type = "transformer" # Using LRA's Transformer model

    # These are the crucial model architecture parameters
    config.model.num_layers = 6            # nanoGPT n_layer
    config.model.num_heads = 6             # nanoGPT n_head
    config.model.emb_dim = 384             # nanoGPT n_embd
    config.model.mlp_dim = 1536            # nanoGPT MLP dim (often 4 * emb_dim)
    config.model.qkv_dim = config.model.emb_dim // config.model.num_heads # Standard practice
    config.model.dropout_rate = 0.1        # nanoGPT dropout
    
    # You might need to adjust learning rate and batch size for a larger model
    # The base_pathfinder32_config likely has defaults, but you can override
    config.learning_rate = 0.001 # Or whatever was good for nanoGPT, adjust if needed
    config.batch_size = 32       # Pathfinder32 can be memory intensive. Adjust if OOM.
                                 # Original transformer_base.py had batch_size = 256,
                                 # but emb_dim was only 32. With emb_dim=384, you'll need smaller batch.

    # Other parameters from base_pathfinder32_config are usually fine
    # config.attention_dropout_rate = 0.1 # if you want separate attention dropout

    return config

def get_hyper(hyper):
    return hyper.product([]) # No hyperparameter sweep for this example