# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import pdb

import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer_task import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(5, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(5, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        ## Encoder input
       
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(5, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(5, hidden_dim)  # project qpos to embedding

        ## Encoder Output
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer('pos_table',
                             get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))  # [CLS], qpos, a_seq
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding

        # decoder parameters
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        
        num_tasks = 5
        self.task_id_proj = nn.Embedding(num_tasks, hidden_dim) #num_tasks = # TODO
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.additional_pos_embed = nn.Embedding(3, hidden_dim)  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_id=None, split=None, **kwargs):
        """
        task_id: (batch, 1)
        qpos: (batch, qpos_dim)
        image: (batch, num_cam, channel, height, width)
        env_state: None
        actions: (batch, seq, action_dim)
        is_pad: (batch, seq)
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # print(is_training)
        # Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+2, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
        
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed,
                                          src_key_padding_mask=is_pad)  # (seq+2, bs, hidden_dim)
            encoder_output = encoder_output[0]  # take cls output only, (bs, hidden_dim)
            latent_info = self.latent_proj(encoder_output)  # (bs, 2*latent_dim)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)  # (bs, latent_dim)
            latent_input = self.latent_out_proj(latent_sample)  # (bs, hidden_dim)
    
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature, (b, c, h, w)
                pos = pos[0]  # (1, c, h, w)
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)  # (b, hidden_dim)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)  # (b, c, h, w)
            pos = torch.cat(all_cam_pos, axis=3)  # (1, c, h, w)

            if split == None:
                #print(task_id)
                taskid_embed = self.task_id_proj(task_id)
                #print(taskid_embed.shape)
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                  self.additional_pos_embed.weight, taskid_embed = taskid_embed)[0]
                a_hat = self.action_head(hs)  # (b, seq, 5)
                is_pad_hat = self.is_pad_head(hs)  # (b, seq, 1)

                return a_hat, is_pad_hat, [mu, logvar]    

            
            if split=="Decoder":
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                    self.additional_pos_embed.weight, split=split, **kwargs)[0]
                
                a_hat = self.action_head(hs)  # (b, seq, 5)
                
                is_pad_hat = self.is_pad_head(hs)  # (b, seq, 1)
                
                return a_hat, is_pad_hat, [mu, logvar]


            elif split=="Encoder":
                taskid_embed = self.task_id_proj(torch.unsqueeze(task_id.float(), axis=0))
                # print(task_id)
                memory, tgt = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                    self.additional_pos_embed.weight,  taskid_embed = taskid_embed, split=split)              

                return memory, tgt


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 5  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model

