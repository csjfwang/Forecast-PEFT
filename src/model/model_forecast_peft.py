from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# For prompt init
import math
from functools import reduce
from operator import mul

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder
from .layers.transformer_blocks import Block
from .layers.transformer_blocks_peft import BlockPEFT

class ModelPEFT(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        param_efficient=True,
        prompt_length=50,
        adapter_rank=64,
        modes=6,
    ) -> None:
        super().__init__()
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        # Forecast-FT or Forecast-PEFT
        if param_efficient:
            self.blocks = nn.ModuleList(
                BlockPEFT(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    adapter_rank=adapter_rank,
                )
                for i in range(encoder_depth)
            )
        else:
            self.blocks = nn.ModuleList(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                )
                for i in range(encoder_depth)
            )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        # Using decoder pretrained by MAE
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        decoder_depth = 4
        self.embed_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        if param_efficient:
            self.decoder_blocks = nn.ModuleList(
                BlockPEFT(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    adapter_rank=adapter_rank,
                )
                for i in range(decoder_depth)
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                )
                for i in range(decoder_depth)
            )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.future_pred = nn.Linear(embed_dim, future_steps * 2)

        self.future_steps = future_steps

        # Pi Head:
        self.prompt_pi = nn.Linear(embed_dim, 1)

        # Modality Control Prompts (MCPs): for the original decoder inputs
        self.num_tokens_mcp = prompt_length
        self.mode_num = modes
        self.mcp_prompt_embeddings = nn.Parameter(torch.zeros(
            self.mode_num, self.num_tokens_mcp, embed_dim))
        # xavier_uniform initialization
        val = math.sqrt(6. / float(3 * reduce(mul, (self.num_tokens_mcp, 1), 1) + embed_dim)) 
        nn.init.uniform_(self.mcp_prompt_embeddings.data, -val, val)

        # Contextural Embedding Prompts (CEPs): for the encoder inputs
        self.num_tokens_cep = prompt_length
        self.cep_num = encoder_depth 
        self.cep_prompt_embeddings = nn.Parameter(torch.zeros(
            self.cep_num, self.num_tokens_cep, embed_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.cep_prompt_embeddings.data, -val, val)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, 49], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        # Insert CEPs from the Input side
        self.prompt_mask_input = torch.zeros(1, self.num_tokens_cep, dtype=torch.bool, device=key_padding_mask.device)

        x_encoder = torch.cat((
                x_encoder[:, :N, :],
                self.cep_prompt_embeddings[0].expand(B, -1, -1),
                x_encoder[:, N:, :]
            ), dim=1)
        key_padding_mask = torch.cat((
                key_padding_mask[:, :N],
                self.prompt_mask_input.expand(B, -1),
                key_padding_mask[:, N:]
            ), dim=1)
        index_deep_prompt = 1

        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
            if index_deep_prompt <= self.cep_prompt_embeddings.shape[0] - 1:
                x_encoder = torch.cat((
                    x_encoder[:, :N, :],
                    self.cep_prompt_embeddings[index_deep_prompt].expand(B, -1, -1),
                    x_encoder[:, (N+self.num_tokens_cep):, :]
                ), dim=1)
            index_deep_prompt += 1
        x_encoder = self.norm(x_encoder)

        # Using original decoder pretrained by MAE
        x_decoder = self.decoder_embed(x_encoder)
        assert x_decoder.shape[1] == N + self.num_tokens_cep + M
        hist_tokens = x_decoder[:, :N]
        lane_tokens = x_decoder[:, -M:]
        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        
        # pos feature
        x_centers = torch.cat(
            [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
        )
        angles = torch.cat(
            [
                data["x_angles"][..., 49],
                data["x_angles"][..., 49],
                data["lane_angles"],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)

        x_decoder = torch.cat(
            [hist_tokens, 
             decoder_fut_token, 
             lane_tokens], 
             dim=1)
        
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat)

        # Insert MCPs for multi modal outputs
        self.prompt_mask = torch.zeros(1, self.num_tokens_mcp, dtype=torch.bool, device=key_padding_mask.device)

        future_padding_mask = data["x_padding_mask"][:, :, 50:]
        decoder_key_padding_mask = torch.cat(
            [
                data["x_key_padding_mask"],
                self.prompt_mask.expand(B, -1),
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        x_decoder_tmp = torch.cat(
            [x_decoder[:, :N], 
             self.mcp_prompt_embeddings[0].expand(B, -1, -1),
             x_decoder[:, N:]], 
             dim=1)

        for blk in self.decoder_blocks:
            x_decoder_tmp = blk(x_decoder_tmp, key_padding_mask=decoder_key_padding_mask)
        
        x_decoder_tmp = self.decoder_norm(x_decoder_tmp)
        hist_token = x_decoder_tmp[:, :N].reshape(-1, self.embed_dim)
        future_token = x_decoder_tmp[:, N + self.num_tokens_mcp : (2 * N) + self.num_tokens_mcp].reshape(-1, self.embed_dim)
        lane_token = x_decoder_tmp[:, -M:]

        # The first mode: 
        y_hat_all = self.future_pred(future_token).view(B, -1, self.future_steps, 2)
        y_hat = y_hat_all[:,:1]
        y_hat_others = y_hat_all[:,1:]
        pi_token = x_decoder_tmp[:, N:N+1].reshape(-1, self.embed_dim)
        pi = self.prompt_pi(pi_token).view(B, -1)

        # The rest (mode_num - 1) modes:
        for index in range(self.mode_num - 1):
            x_decoder_tmp = torch.cat(
                [x_decoder[:, :N], 
                self.mcp_prompt_embeddings[index+1].expand(B, -1, -1),
                x_decoder[:, N:]], 
                dim=1)

            for blk in self.decoder_blocks:
                x_decoder_tmp = blk(x_decoder_tmp, key_padding_mask=decoder_key_padding_mask)
            
            x_decoder_tmp = self.decoder_norm(x_decoder_tmp)
            future_token = x_decoder_tmp[:, N + self.num_tokens_mcp : (2 * N) + self.num_tokens_mcp].reshape(-1, self.embed_dim)
            
            y_hat_all = self.future_pred(future_token).view(B, -1, self.future_steps, 2)
            y_hat = torch.cat([y_hat, y_hat_all[:,:1]], dim=1)
            pi_token = x_decoder_tmp[:, N:N+1].reshape(-1, self.embed_dim)
            pi = torch.cat([pi, self.prompt_pi(pi_token).view(B, -1)], dim=1)
        
        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
        }
