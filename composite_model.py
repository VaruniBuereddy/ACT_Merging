from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG # must import first
import torch
import torch.nn as nn
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

from training.utils_v2 import *


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='merged_dataset')
args = parser.parse_args()
task = args.task

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)

# device
device = 'cuda'#os.environ['DEVICE']

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, task_id = data
    image_data, qpos_data, action_data, is_pad, task_id = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device), task_id.to(device)
    return policy(qpos_data, image_data, action_data, is_pad, task_id) # TODO remove None

class CompositeNetwork(nn.Module):
    def __init__(self, model1, model2, intermediate_dim=256):
        super(CompositeNetwork, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.intermediate_layers = nn.Sequential(
            nn.Linear(512, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 512)
        )
        # self.intermediate_ffn = nn.Sequential(
        #     nn.Linear(512, intermediate_dim),
        #     nn.ReLU(),
        #     nn.Linear(intermediate_dim, intermediate_dim),
        #     nn.ReLU(),
        #     nn.Linear(intermediate_dim, 512)
        # )


    def forward(self, *x):
        mem, tgt = self.model1(*x, split = 'Encoder')
        x1 = self.intermediate_layers(mem)
        fwd_dict = self.model2(*x, split = 'Decoder', **{'memory':x1, 'tgt':tgt})

        return fwd_dict


## Configuration Area
dataset1_dir = '/scratch/gb2643/documents/ACT/data/close_drawer'
dataset2_dir = '/scratch/gb2643/documents/ACT/data/pick_goal@30'
checkpoint1_path = '/scratch/gb2643/documents/ACT_MERGE/checkpoints/close_drawer'
checkpoint2_path = '/scratch/gb2643/documents/ACT_MERGE/checkpoints/pick_goal@30'

task_ids = [1, 1000]
checkpoint_dir = 'merged_ckpts/Linear'
os.makedirs(checkpoint_dir, exist_ok=True)

#################


policy1 = make_policy(policy_config['policy_class'], policy_config)
policy1.load_state_dict(torch.load(os.path.join(checkpoint1_path, 'policy_last.ckpt')))
policy2 = make_policy(policy_config['policy_class'], policy_config)
policy2.load_state_dict(torch.load(os.path.join(checkpoint2_path, 'policy_last.ckpt')))


policy1.to(device)
policy2.to(device)

for param in policy1.parameters():
    param.requires_grad = False

for param in policy2.parameters():
    param.requires_grad = False


composite_policy = CompositeNetwork(policy1, policy2)
composite_policy.to(device)

for name, param in composite_policy.named_parameters():
    if param.requires_grad:
        print(f'Name: {name}')
        # print(f'Parameter:\n{param.data}\n')


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, composite_policy.parameters()), lr=POLICY_CONFIG['lr'], weight_decay=0.01)
train_dataloader, val_dataloader, stats, _ = load_datasets(dataset1_dir, dataset2_dir, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'], task_ids=task_ids)


stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')


with open(stats_path, 'wb') as f:
    pickle.dump(stats, f)

train_history = []
validation_history = []
min_val_loss = np.inf
best_ckpt_info = None
for epoch in range(train_cfg['num_epochs']):
    print(f'\nEpoch {epoch}')
    # validation
    with torch.inference_mode():
        composite_policy.eval()
        epoch_dicts = []
        for batch_idx, data in enumerate(val_dataloader):
            forward_dict = forward_pass(data, composite_policy)
            epoch_dicts.append(forward_dict)
        epoch_summary = compute_dict_mean(epoch_dicts)
        validation_history.append(epoch_summary)

        epoch_val_loss = epoch_summary['loss']
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_ckpt_info = (epoch, min_val_loss, deepcopy(composite_policy.state_dict()))
    print(f'Val loss:   {epoch_val_loss:.5f}')
    summary_string = ''
    for k, v in epoch_summary.items():
        summary_string += f'{k}: {v.item():.3f} '
    print(summary_string)

    # training
    composite_policy.train()
    optimizer.zero_grad()
    for batch_idx, data in enumerate(train_dataloader):
        forward_dict = forward_pass(data, composite_policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_history.append(detach_dict(forward_dict))
    epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
    epoch_train_loss = epoch_summary['loss']
    print(f'Train loss: {epoch_train_loss:.5f}')
    summary_string = ''
    for k, v in epoch_summary.items():
        summary_string += f'{k}: {v.item():.3f} '
    print(summary_string)

    if epoch % 200 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
        #torch.save(composite_policy.state_dict(), ckpt_path)
        plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
torch.save(composite_policy.state_dict(), ckpt_path)
