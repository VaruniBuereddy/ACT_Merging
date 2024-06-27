import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = '/home/varuni/Documents/Researchwork/act/data'

# checkpoint directory
CHECKPOINT_DIR = '/home/varuni/Documents/Researchwork/act/checkpoints/task_id_1_1000/close_drawer'

# device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    # Set the default device for PyTorch tensors to CUDA device
    torch.cuda.set_device(0)  # Set to the specific GPU device index if you have multiple GPUs

os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/ttyACM1',
    'follower': '/dev/ttyACM0'
     
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 1000,
    'state_dim': 5,
    'action_dim': 5,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 50,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    # 'policy_class': 'EMB',
    'temporal_agg': False,
    'optimizer': 'Adam',
    'scheduler': 'step',
    'step_size': 1000,
    'gamma': 1.0,
    'sample_mode': 'dec',     # inc, dec, random
    'sample_schedule': 'sigmoid', # linear, cosine, sigmoid
    'l2_weight': 0.01,
    'nlogp_weight': 0.01
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 5000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}
