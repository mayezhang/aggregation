import argparse
import os
import sys


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.dynaggr_dataset import DEFAULT_DATASET_PATH, RAW_DATA_ROOT, build_offline_dataset


def parse_args():
    parser = argparse.ArgumentParser("Build offline dataset for dynamic aggregation learning")
    parser.add_argument("--raw_data_root", type=str, default=RAW_DATA_ROOT, help="root directory that contains dfx_data.pkl files")
    parser.add_argument("--output_path", type=str, default=DEFAULT_DATASET_PATH, help="where to store the processed dataset")
    parser.add_argument("--tail_penalty", type=float, default=0.0, help="optional reward penalty for tail errors")
    parser.add_argument("--per_high_cutoff", type=float, default=90.0, help="drop samples with PER above this threshold")
    parser.add_argument("--min_tx_ppdu_time_us", type=float, default=600.0, help="drop samples with tx_ppdu_time below this threshold")
    parser.add_argument(
        "--min_occupancy_for_update",
        type=float,
        default=0.0,
        help="optional low-occupancy filter: drop samples with tx_ppdu_time/real_ppdu_time_lim below this threshold",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="divide reward by this scale before learning",
    )
    parser.add_argument(
        "--state_reward_excel_path",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "state_reward_debug.xlsx"),
        help="debug Excel path: one sheet per dfx_data.pkl with processed state/reward",
    )
    parser.add_argument("--disable_state_reward_excel", action="store_true", help="disable debug Excel export")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = build_offline_dataset(
        data_root=os.path.abspath(args.raw_data_root),
        output_path=os.path.abspath(args.output_path),
        tail_penalty=args.tail_penalty,
        per_high_cutoff=args.per_high_cutoff,
        min_tx_ppdu_time_us=args.min_tx_ppdu_time_us,
        min_occupancy_for_update=args.min_occupancy_for_update,
        reward_scale=args.reward_scale,
        debug_excel_path=None if args.disable_state_reward_excel else os.path.abspath(args.state_reward_excel_path),
    )
    summary = dataset["summary"]
    print(
        "dataset prepared:",
        f"transitions={summary['num_transitions']},",
        f"states={summary['num_states']},",
        f"episodes={summary['num_episodes']},",
        f"coarse_states={summary['num_coarse_states']},",
        f"kept_ratio={summary['sample_keep_ratio']:.3f},",
        f"default_action={summary['default_action_value']},",
        f"behavior_reward={summary['behavior_reward_mean']:.4f}",
    )
