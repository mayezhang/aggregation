#%%
import pickle
import numpy as np
import random

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

data_path = './data/dynaggr_offline_dataset.pkl'

loaded = load_pkl(data_path)

loaded['episode_slices']

rx_mac_episode_groups = loaded['rx_mac_episode_groups']

rx_macs = list(rx_mac_episode_groups.keys())
random.shuffle(rx_macs)
ordered_episode_ids = []
for rx_mac in rx_macs:
    episode_ids = list(rx_mac_episode_groups[rx_mac])
    random.shuffle(episode_ids)
    ordered_episode_ids.extend(episode_ids)
rx_macs, ordered_episode_ids

#%%

loaded['state_visit_counts']