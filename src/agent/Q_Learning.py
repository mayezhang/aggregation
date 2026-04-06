import numpy as np
from collections import defaultdict
import os, sys
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_base import ModelBase
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'ai_dynaggr_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 核心修复

class QLearning:
    def __init__(self, args):
        self.agent_name = args.algo_name
        self.lr = args.lr
        self.min_lr = min(args.min_lr, args.lr)
        self.gamma = args.gamma
        self.action_dim = args.action_dim
        self.conservative_coef = args.conservative_coef
        self.min_support = args.min_support
        self.default_action_idx = args.default_action_idx
        # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.float64))
        self.visit_table = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.int32))

    def _masked_q_values(self, state, state_action_counts):
        masked_q = np.full(self.action_dim, -np.inf, dtype=np.float64)
        if state_action_counts is None:
            return masked_q

        counts = np.asarray(state_action_counts, dtype=np.float64)
        supported = counts > 0
        if not np.any(supported):
            return masked_q

        eligible = counts >= self.min_support
        if not np.any(eligible):
            eligible = supported

        q_values = self.Q_table[state].copy()
        penalties = np.zeros(self.action_dim, dtype=np.float64)
        penalties[eligible] = self.conservative_coef / np.sqrt(np.maximum(counts[eligible], 1.0))
        masked_q[eligible] = q_values[eligible] - penalties[eligible]
        return masked_q

    def greedy_action(self, state, state_action_counts=None):
        masked_q = self._masked_q_values(state, state_action_counts)
        if np.all(np.isneginf(masked_q)):
            return self.default_action_idx
        return int(np.argmax(masked_q))

    def update(self, state, action, reward, next_state, done, next_state_counts=None):
        """ 策略更新函数，这里实现伪代码中的更新公式 """
        Q_predict = self.Q_table[state][action]
        if done:
            next_state_value = 0.0
        else:
            masked_q = self._masked_q_values(next_state, next_state_counts)
            next_state_value = 0.0 if np.all(np.isneginf(masked_q)) else float(np.max(masked_q))

        Q_target = reward + self.gamma * next_state_value
        self.visit_table[state][action] += 1
        visit_count = int(self.visit_table[state][action])
        adaptive_lr = max(self.min_lr, self.lr / np.sqrt(max(visit_count, 1)))
        # 访问次数越多，学习率越小，让表格估值后期更稳定。
        self.Q_table[state][action] += adaptive_lr * (Q_target - Q_predict)
        return np.abs(Q_target - Q_predict), adaptive_lr

class QLearningModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = QLearning(args)  # cfg存储算法相关参数
        self.model_name = f'{self.agent.agent_name}_num_{1}_seed_{self.args.seed}'
 
    def train(self):
        history = {'td_err': [], 'reward': [], 'avg_q': [], 'policy_reward': [], 'lr': []}
        for epoch in range(self.args.epochs):
            # 一个 epoch 会扫过当前重排后的所有 episode。
            episode_reward = []
            episode_loss = []
            episode_avg_q = []
            episode_lr = []
            self.env.reset()
            while True:
                state, action, reward, next_state, done = self.env.step()
                if state is None:
                    break
                next_state_counts = None if done else self.env.get_state_action_counts(next_state)
                td_err, used_lr = self.agent.update(state, action, reward, next_state, done, next_state_counts)

                episode_reward.append(reward)
                episode_loss.append(td_err)
                episode_avg_q.append(np.max(self.agent.Q_table[state]))
                episode_lr.append(used_lr)

            policy_reward = self.estimate_policy_reward()
            mean_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
            mean_avg_q = float(np.mean(episode_avg_q)) if episode_avg_q else 0.0
            mean_reward = float(np.mean(episode_reward)) if episode_reward else 0.0
            mean_lr = float(np.mean(episode_lr)) if episode_lr else self.agent.min_lr
            if epoch % self.args.log_interval == 0 or epoch == self.args.epochs - 1:
                print(
                    f"epoch {epoch} "
                    f"loss {mean_loss:.4f}, "
                    f"avg_q {mean_avg_q:.4f}, "
                    f"avg_lr {mean_lr:.5f}, "
                    f"policy_reward {policy_reward:.4f}"
                )
            history['td_err'].append(mean_loss)
            history['reward'].append(mean_reward)
            history['avg_q'].append(mean_avg_q)
            history['policy_reward'].append(policy_reward)
            history['lr'].append(mean_lr)
        self.plot(history)
        self.plot_dataset_statistics()
        self.save_policy()
        print(f"完成训练！Q-Table shape: {len(self.agent.Q_table), self.agent.action_dim}")

    def plot(self, history):
        epochs = range(len(history['td_err']))
        plt.figure(figsize=(18, 5))
        # 1. TD Error 曲线 (判断收敛性)
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history['td_err'], color='red')
        plt.title('TD Error (Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid(True)

        # 2. 平均 Q 值曲线 (判断估值稳定性)
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history['avg_q'], color='blue')
        plt.title('Mean Max Q-Value')
        plt.xlabel('Epochs')
        plt.ylabel('Q Value')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['policy_reward'], color='green')
        plt.title('Estimated Policy Reward')
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.grid(True)

        os.makedirs(os.path.dirname(self.args.plot_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.args.plot_path, dpi=200)
        plt.close()

    def _top_counter_items(self, counter_like, top_k=20):
        if not counter_like:
            return []
        if hasattr(counter_like, "items"):
            items = list(counter_like.items())
        else:
            items = list(counter_like)
        items.sort(key=lambda item: item[1], reverse=True)
        return items[:top_k]

    def _format_state_label(self, state, max_len=42):
        label = str(state)
        if len(label) <= max_len:
            return label
        return label[: max_len - 3] + "..."

    def plot_dataset_statistics(self):
        state_visit_counts = self.env.dataset_meta.get('state_visit_counts', {})
        state_transition_counts = self.env.dataset_meta.get('state_transition_counts', {})
        global_action_counts = np.asarray(
            self.env.dataset_meta.get('global_action_counts', np.zeros(self.args.action_dim)),
            dtype=np.int32,
        )

        top_states = self._top_counter_items(state_visit_counts, top_k=15)
        top_transitions = self._top_counter_items(state_transition_counts, top_k=15)
        if not top_states and not top_transitions and not np.any(global_action_counts):
            return

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        if top_states:
            labels = [self._format_state_label(item[0]) for item in top_states]
            values = [int(item[1]) for item in top_states]
            plt.barh(labels[::-1], values[::-1], color='#4472c4')
        plt.title('Top State Visits')
        plt.xlabel('Count')
        plt.grid(True, axis='x', alpha=0.3)

        plt.subplot(1, 3, 2)
        if top_transitions:
            labels = [
                f"{self._format_state_label(src, 18)} -> {self._format_state_label(dst, 18)}"
                for (src, dst), _ in top_transitions
            ]
            values = [int(item[1]) for item in top_transitions]
            plt.barh(labels[::-1], values[::-1], color='#70ad47')
        plt.title('Top State Transitions')
        plt.xlabel('Count')
        plt.grid(True, axis='x', alpha=0.3)

        plt.subplot(1, 3, 3)
        action_labels = [str(action_value) for action_value in self.env.action_values]
        plt.bar(action_labels, global_action_counts, color='#ed7d31')
        plt.title('Action Coverage')
        plt.xlabel('PPDU Limit (us)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)

        os.makedirs(os.path.dirname(self.args.transition_plot_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.args.transition_plot_path, dpi=200)
        plt.close()

    def estimate_policy_reward(self):
        total_reward = 0.0
        total_count = 0
        for state, counts in self.env.state_action_counts.items():
            counts = np.asarray(counts, dtype=np.int32)
            if counts.sum() == 0:
                continue
            mean_rewards = self.env.get_state_action_mean_rewards(state)
            action = self.agent.greedy_action(state, counts)
            total_reward += float(mean_rewards[action]) * int(counts.sum())
            total_count += int(counts.sum())
        if total_count == 0:
            return 0.0
        return total_reward / total_count

    def save_policy(self):
        policy_table = {}
        for state, counts in self.env.state_action_counts.items():
            policy_table[state] = self.agent.greedy_action(state, counts)

        coarse_policy_table = {}
        for coarse_state, counts in self.env.coarse_state_action_counts.items():
            counts = np.asarray(counts, dtype=np.int32)
            if counts.sum() == 0:
                continue
            reward_sum = np.asarray(self.env.coarse_state_action_reward_sum[coarse_state], dtype=np.float64)
            mean_rewards = reward_sum / np.maximum(counts, 1)
            eligible = counts >= self.args.min_support
            if not np.any(eligible):
                eligible = counts > 0
            scores = np.full(self.args.action_dim, -np.inf, dtype=np.float64)
            scores[eligible] = mean_rewards[eligible]
            coarse_policy_table[coarse_state] = int(np.argmax(scores))

        artifact = {
            'action_values': self.env.action_values,
            'feature_names': self.env.feature_names,
            'default_action_idx': self.args.default_action_idx,
            'default_action_value': self.env.get_action_value(self.args.default_action_idx),
            'policy_table': policy_table,
            'coarse_policy_table': coarse_policy_table,
            'q_table': {state: values.copy() for state, values in self.agent.Q_table.items()},
            'state_action_counts': self.env.state_action_counts,
            'summary': {
                **self.env.summary,
                'estimated_policy_reward': self.estimate_policy_reward(),
                'num_q_states': len(self.agent.Q_table),
                'min_support': self.args.min_support,
                'conservative_coef': self.args.conservative_coef,
                'lr': self.args.lr,
                'min_lr': self.args.min_lr,
            },
        }

        os.makedirs(os.path.dirname(self.args.policy_path), exist_ok=True)
        with open(self.args.policy_path, 'wb') as file_obj:
            import pickle
            pickle.dump(artifact, file_obj)
