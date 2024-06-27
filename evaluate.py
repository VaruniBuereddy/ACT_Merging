from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS, CHECKPOINT_DIR # must import first

import os
import cv2
import torch
import pickle
import argparse
from time import time

from robot import Robot
from training.utils import *


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='close_drawer')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
checkpoint_dir = CHECKPOINT_DIR
device = os.environ['DEVICE']


def capture_image(cam):
    # Capture a single frame
    _, frame = cam.read()
    # Generate a unique filename with the current date and time
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Define your crop coordinates (top left corner and bottom right corner)
    x1, y1 = 400, 0  # Example starting coordinates (top left of the crop rectangle)
    x2, y2 = 1600, 900  # Example ending coordinates (bottom right of the crop rectangle)
    # Crop the image
   # image = image[y1:y2, x1:x2]
    # Resize the image
    image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)

    return image

if __name__ == "__main__":
    task_dict = {'close_drawer': [100], 'pick_goal':[100]}
    task_id = task_dict[args.task]
    task_id = torch.from_numpy(np.array(task_id))

    # task_id = torch.unsqueeze(task_id, 0)
    task_id = task_id.to(device)
    # init camera
    cam = cv2.VideoCapture(cfg['camera_port'])
    # Check if the camera opened successfully
    if not cam.isOpened():
        raise IOError("Cannot open camera")
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    # init follower
    follower = Robot(device_name=ROBOT_PORTS['follower'])

    # load the policy
    #checkpoint_dir = CHECKPOIN#os.path.join('checkpoints') #newly added by me
    ckpt_path = os.path.join(checkpoint_dir, train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    #checkpoint_dir = os.path.join(checkpoint_dir, task)
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    print(f'Loaded: {stats_path}')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # bring the follower to the leader
    for i in range(90):
        follower.read_position()
        _ = capture_image(cam)
    
    obs = {
        'qpos': pwm2pos(follower.read_position()),
        'qvel': vel2pwm(follower.read_velocity()),
        'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
    }
    os.system('espeak "start"')

    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            for t in range(cfg['episode_len']):
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image, task_id=task_id)
                    print(all_actions)
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # corrected_index = t % all_actions.shape[1]
                    raw_action = all_actions[:, t % query_frequency]
                    # print("Shape of all_actions:", all_actions.shape)
                    # print("Value of t:", t)
                    # print("Value of query_frequency:", query_frequency)
                    # print("t % query_frequency:", t % query_frequency)


                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                
                action = pos2pwm(action).astype(int)
                #print("action",action)
                ### take action
                follower.set_goal_pos(action)

                ### update obs
                obs = {
                    'qpos': pwm2pos(follower.read_position()),
                    'qvel': vel2pwm(follower.read_velocity()),
                    'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)

        os.system('espeak "ahh"')

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        t0 = time()
        max_timesteps = len(data_dict['/observations/qpos'])
        # create data dir if it doesn't exist
        data_dir = cfg['dataset_dir']  
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')
        # save the data
        with h5py.File("data/demo/trained.hdf5", 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
    
    # disable torque
    follower._disable_torque()