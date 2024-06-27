"""
Add energy-based model for regression (ECCV 2020)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_EMB_model_and_optimizer
from training.emb_utils import stds, num_samples, sample_gmm

# import IPython
# e = IPython.embed

import pdb


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """
        :param qpos: (b, 5,)
        :param image: (b, 1, 3, h, w)
        :param actions: (b, ep_len, 5)
        :param is_pad: (b, ep_len)
        :return:
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, _, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior

            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class EMBPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, reg_optimizer, embed_emb_optimizer = build_EMB_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.reg_optimizer = reg_optimizer
        self.embed_emb_optimizer = embed_emb_optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, compute_score=False, sample_mode='random', ratio=0.0):
        """
        :param qpos: (b, 5,)
        :param image: (b, 1, 3, h, w)
        :param actions: (b, ep_len, 5)
        :param is_pad: (b, ep_len)
        :return:
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.reg_net.num_queries]
            is_pad = is_pad[:, :self.model.reg_net.num_queries]
            loss_dict = dict()
            if compute_score:
                score_gt = self.model(qpos, image, env_state, actions, is_pad, compute_score)   # (b, 1)
                score_gt = score_gt.squeeze(1)  # (b,)

                action_samples, log_action_probs = sample_gmm(actions, stds, num_samples, sample_mode, ratio)

                qpos_samples = qpos.unsqueeze(1).repeat(1, num_samples, 1).view(-1, qpos.shape[-1])
                is_pad_samples = is_pad.unsqueeze(1).repeat(1, num_samples, 1).view(-1, is_pad.shape[-1])
                action_samples = action_samples.view(-1, *action_samples.shape[2:])

                score_samples = self.model(qpos_samples, image, env_state, action_samples, is_pad_samples, compute_score)
                score_samples = score_samples.view(qpos.shape[0], -1)   # (b, num_samples)0

                # nlogp = torch.mean(torch.exp(torch.log(torch.mean(torch.exp(score_samples - log_action_probs), dim=1)) - score_gt))
                alpha = 1
                nlogp = torch.mean(torch.mean(torch.exp(alpha * (score_samples - log_action_probs)), dim=1) / torch.exp(score_gt))
                # nlogp = torch.mean(torch.mean(torch.cat([torch.exp(alpha * (score_samples - log_action_probs)), torch.exp(score_gt).unsqueeze(1)], dim=1), dim=1) / torch.exp(score_gt))

                loss_dict['nlogp'] = nlogp
            else:
                a_hat, _, (mu, logvar), score_hat = self.model(qpos, image, env_state, actions, is_pad, compute_score)

                alpha = 1
                nscore = torch.exp(-alpha * score_hat).mean()
                loss_dict['nscore'] = nscore

                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]

                # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
                loss_dict['loss'] = loss_dict['nscore'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
        else:  # inference time
            a_hat, _, (_, _), _ = self.model(qpos, image, env_state)  # no action, sample from prior

            return a_hat

    def configure_optimizers(self):
        return self.reg_optimizer, self.embed_emb_optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
