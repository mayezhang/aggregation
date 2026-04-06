import os
import sys
import random


# 确保可以导入 learning_code 下的 src
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.agent.Q_Learning import QLearningModel
from src.utils.dynaggr_dataset import DEFAULT_DATASET_PATH, RAW_DATA_ROOT, build_offline_dataset
from src.utils.wifi_env import WifiOfflineEnv
import argparse
import numpy as np


def args():
    parser = argparse.ArgumentParser("Setting for AI-DYNAGGR")
    parser.add_argument("--algo_name", type=str, default="q-learning", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    # Training Params
    parser.add_argument("--epochs", type=int, default=80, help="epochs")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate of Q-table")
    parser.add_argument("--min_lr", type=float, default=5e-3, help="Minimum adaptive learning rate of Q-table")
    parser.add_argument("--gamma", type=float, default=0.65, help="Discount factor")
    parser.add_argument("--min_support", type=int, default=3, help="只在样本数不少于该阈值的动作上做保守贪心")
    parser.add_argument("--conservative_coef", type=float, default=0.8, help="对低支持动作的保守惩罚系数")
    parser.add_argument("--log_interval", type=int, default=5, help="日志打印间隔")

    parser.add_argument("--data_path", type=str, default=DEFAULT_DATASET_PATH, help="processed dataset path")
    parser.add_argument("--raw_data_root", type=str, default=RAW_DATA_ROOT, help="raw dfx_data.pkl root")
    parser.add_argument("--rebuild_dataset", action="store_true", help="rebuild processed dataset before training")
    parser.add_argument("--tail_penalty", type=float, default=0.0, help="optional tail error penalty added to reward")
    parser.add_argument("--per_high_cutoff", type=float, default=90.0, help="drop samples with PER above this threshold")
    parser.add_argument("--min_tx_ppdu_time_us", type=float, default=600.0, help="drop samples with tx_ppdu_time below this threshold")
    parser.add_argument(
        "--min_occupancy_for_update",
        type=float,
        default=0.0,
        help="optional low-occupancy filter: drop samples with tx_ppdu_time/real_ppdu_time_lim below this threshold",
    )
    parser.add_argument("--reward_scale", type=float, default=1.0, help="divide reward by this scale before learning")
    parser.add_argument(
        "--state_reward_excel_path",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "state_reward_debug.xlsx"),
        help="debug Excel path: one sheet per dfx_data.pkl with processed state/reward",
    )
    parser.add_argument("--disable_state_reward_excel", action="store_true", help="disable debug Excel export")
    parser.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(ROOT_DIR, "artifacts", "training_curve.png"),
        help="path to save training curves",
    )
    parser.add_argument(
        "--transition_plot_path",
        type=str,
        default=os.path.join(ROOT_DIR, "artifacts", "dataset_transition_stats.png"),
        help="path to save dataset/state transition statistics",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default=os.path.join(ROOT_DIR, "artifacts", "dynaggr_policy.pkl"),
        help="path to save learned policy",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dataset(args):
    if args.rebuild_dataset or not os.path.exists(args.data_path):
        dataset = build_offline_dataset(
            data_root=args.raw_data_root,
            output_path=args.data_path,
            tail_penalty=args.tail_penalty,
            per_high_cutoff=args.per_high_cutoff,
            min_tx_ppdu_time_us=args.min_tx_ppdu_time_us,
            min_occupancy_for_update=args.min_occupancy_for_update,
            reward_scale=args.reward_scale,
            debug_excel_path=None if args.disable_state_reward_excel else args.state_reward_excel_path,
        )
        summary = dataset["summary"]
        print(
            "dataset prepared:",
            f"transitions={summary['num_transitions']},",
            f"states={summary['num_states']},",
            f"episodes={summary['num_episodes']},",
            f"kept_ratio={summary['sample_keep_ratio']:.3f},",
            f"default_action={summary['default_action_value']}",
        )
    return args.data_path


def make_env(args):
    """ 配置环境 """
    env = WifiOfflineEnv(args.data_path)  # 定义环境

    state_dim = env.state_dim  # 状态数
    action_dim = env.action_dim # 动作数

    if state_dim is None or action_dim is None:
        raise ValueError("state/action dim is err.")

    # print(f"state dim:{state_dim}, action dim:{action_dim}")
    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'default_action_idx', env.default_action_idx)

    return env


if __name__ == '__main__':
    args = args()
    set_seed(args.seed)
    args.data_path = os.path.abspath(args.data_path)
    args.raw_data_root = os.path.abspath(args.raw_data_root)
    args.state_reward_excel_path = os.path.abspath(args.state_reward_excel_path)
    args.plot_path = os.path.abspath(args.plot_path)
    args.transition_plot_path = os.path.abspath(args.transition_plot_path)
    args.policy_path = os.path.abspath(args.policy_path)
    ensure_dataset(args)
    env = make_env(args)

    if args.algo_name == "q-learning":
        agent = QLearningModel(env=env, args=args)

    agent.train()
