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

    def __call__(self, qpos, image, actions=None, is_pad=None, task_id=None, split=None, **kwargs):
        """
        :Task id: (b, 1)
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
        if actions is not None and split==None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            
            a_hat, _, (mu, logvar) = self.model( qpos, image, env_state, actions, is_pad, task_id=task_id)

            # print(a_hat)
            # print(mu)
            # print(logvar)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
        
        elif actions is not None and split =="Encoder":
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            memory, tgt = self.model(qpos, image, env_state, actions, is_pad, task_id = task_id, split=split)
            return memory, tgt

        elif actions is not None and split =="Decoder":
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, _, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, task_id = task_id,split=split, **kwargs)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
            
            # return a_hat, _, (mu, logvar)


        else:  # inference time

            if(split=='Encoder'):
            # print(f"Task ID: {task_id}")
                mem, tgt =  self.model(qpos, image, env_state, actions, is_pad, task_id = task_id, split=split)
                return mem, tgt

            elif (split=='Decoder'or split==None):
                a_hat, _, (_, _) = self.model(qpos, image, env_state, task_id=task_id, **kwargs)  # no action, sample from prior
                # print(a_hat)    
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
        model, optimizer = build_EMB_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.l2_weight = args_override['l2_weight']
        self.nlogp_weight = args_override['nlogp_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, sample_mode='random', sample_schedule='linear', progress=0.0):
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

            a_hat, _, (mu, logvar), score_gt = self.model(qpos, image, env_state, actions, is_pad)  # (b, 1)

            qpos_samples = sample_gmm(qpos, stds, num_samples, sample_mode, sample_schedule, progress)
            qpos_samples = qpos_samples.squeeze(0)

            _, _, (_, _), score_samples = self.model(qpos_samples, image, env_state, actions, is_pad)  # (b, 1)

            loss_dict = dict()
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]

            # nlogp = torch.mean(score_gt - score_samples)
            nlogp = torch.mean(torch.maximum(score_gt - score_samples + 1, torch.zeros_like(score_gt)))
            loss_dict['nlogp'] = nlogp

            l2 = torch.mean(score_gt ** 2 + score_samples ** 2)
            loss_dict['l2'] = l2

            # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['nlogp'] * self.nlogp_weight + loss_dict['l2'] * self.l2_weight

            return loss_dict
        else:  # inference time
            a_hat, _, (_, _), _ = self.model(qpos, image, env_state)  # no action, sample from prior

            return a_hat

    def configure_optimizers(self):
        return self.optimizer


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
