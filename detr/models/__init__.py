# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
# from .detr_vae import build_cnnmlp as build_cnnmlp
# from .detr_vae import build_emb as build_emb


def build_ACT_model(args):
    return build_vae(args)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)


def build_EMB_model(args):
    return build_emb(args)
