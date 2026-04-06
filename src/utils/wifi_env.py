import random
import pickle
import numpy as np

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_dims(x):
    if isinstance(x, (tuple, list)):
        return len(x)
    elif isinstance(x, np.ndarray):
        return x.shape
    elif np.isscalar(x):
        return 1
    else:
        return None


class WifiOfflineEnv:
    def __init__(self, data_path):
        loaded = load_pkl(data_path)
        if isinstance(loaded, dict) and 'transitions' in loaded:
            self.dataset_meta = loaded
            self.dataset = loaded['transitions']
            self.episode_slices = [tuple(item) for item in loaded.get('episode_slices', [])]
            self.episode_meta = list(loaded.get('episode_meta', []))
            self.rx_mac_episode_groups = {
                str(rx_mac): list(episode_ids)
                for rx_mac, episode_ids in loaded.get('rx_mac_episode_groups', {}).items()
            }
            self.action_values = tuple(loaded['action_values'])
            self.feature_names = tuple(loaded.get('feature_names', ()))
            self.state_action_counts = loaded.get('state_action_counts', {})
            self.state_action_reward_sum = loaded.get('state_action_reward_sum', {})
            self.coarse_state_action_counts = loaded.get('coarse_state_action_counts', {})
            self.coarse_state_action_reward_sum = loaded.get('coarse_state_action_reward_sum', {})
            self.summary = loaded.get('summary', {})
            self.default_action_idx = int(self.summary.get('default_action_idx', 0))
        else:
            self.dataset_meta = {}
            self.dataset = loaded
            self.episode_slices = []
            self.episode_meta = []
            self.rx_mac_episode_groups = {}
            self.action_values = tuple(range(6))
            self.feature_names = tuple()
            self.state_action_counts = {}
            self.state_action_reward_sum = {}
            self.coarse_state_action_counts = {}
            self.coarse_state_action_reward_sum = {}
            self.summary = {}
            self.default_action_idx = 0

        self._ptr = 0
        self.data_size = len(self.dataset)
        self._epoch_transition_indices = list(range(self.data_size))

        state = self.dataset[self._ptr][0]
        self.state_dim = get_dims(state)
        self.action_dim = len(self.action_values)

    def _build_epoch_transition_indices(self):
        """
        Shuffle episode order without breaking the temporal order inside one rx_mac episode.

        We first shuffle episodes inside each rx_mac bucket, then shuffle the rx_mac
        buckets themselves, and finally concatenate all episode slices.
        """
        # 包含每个episode的（start idx, end idx）
        if not self.episode_slices:
            shuffled_indices = list(range(self.data_size))
            random.shuffle(shuffled_indices)
            return shuffled_indices

        # 指示当前的rxmac包含了哪几个episode
        if self.rx_mac_episode_groups:
            rx_macs = list(self.rx_mac_episode_groups.keys())
            random.shuffle(rx_macs)
            ordered_episode_ids = []
            for rx_mac in rx_macs:
                episode_ids = list(self.rx_mac_episode_groups[rx_mac])
                random.shuffle(episode_ids)
                ordered_episode_ids.extend(episode_ids)
        else:
            ordered_episode_ids = list(range(len(self.episode_slices)))
            random.shuffle(ordered_episode_ids)

        epoch_transition_indices = []
        for episode_id in ordered_episode_ids:
            start_idx, end_idx = self.episode_slices[episode_id]
            epoch_transition_indices.extend(range(start_idx, end_idx))
        
        # 返回打乱顺序的每个episode的索引
        return epoch_transition_indices

    def reset(self):
        """重置环境，在每个 epoch 开始时重排 episode 顺序。"""
        self._epoch_transition_indices = self._build_epoch_transition_indices()
        self._ptr = 0
        if not self._epoch_transition_indices:
            return None
        return self.dataset[self._epoch_transition_indices[self._ptr]][0]

    def step(self):
        """模拟 Wi-Fi 环境执行一步"""
        if self._ptr >= len(self._epoch_transition_indices):
            return None, None, None, None, True # 数据读完了
        
        transition_idx = self._epoch_transition_indices[self._ptr]
        transition = self.dataset[transition_idx]
        self._ptr += 1

        if len(transition) == 5:
            state, action, reward, next_state, done = transition
        else:
            state, action, reward, next_state = transition
            done = False
        
        return state, action, reward, next_state, done

    def get_state_action_counts(self, state):
        counts = self.state_action_counts.get(state)
        if counts is None:
            return None
        return np.asarray(counts, dtype=np.int32)

    def get_state_action_mean_rewards(self, state):
        counts = self.get_state_action_counts(state)
        reward_sum = self.state_action_reward_sum.get(state)
        if counts is None or reward_sum is None:
            return None
        reward_sum = np.asarray(reward_sum, dtype=np.float64)
        return reward_sum / np.maximum(counts, 1)

    def get_action_value(self, action_idx):
        return int(self.action_values[action_idx])
