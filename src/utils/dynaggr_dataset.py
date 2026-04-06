import glob
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
LEARNING_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(LEARNING_ROOT, ".."))
RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "data_pre_proc", "data")
DEFAULT_DATASET_PATH = os.path.join(LEARNING_ROOT, "data", "dynaggr_offline_dataset.pkl")


DATA_PRE_PROC_ROOT = os.path.join(PROJECT_ROOT, "data_pre_proc")
if DATA_PRE_PROC_ROOT not in sys.path:
    sys.path.insert(0, DATA_PRE_PROC_ROOT)


ACTION_VALUES = tuple(range(700, 5501, 300))
ACTION_TO_INDEX = {value: index for index, value in enumerate(ACTION_VALUES)}

BW_RATE_NORMALIZE_FACTOR = {
    20: 1.0,
    40: 2.0,
    80: 4.188,
    160: 8.376,
    320: 16.752,
}
BW_FIELD_TO_MHZ = {0: 20, 1: 40, 2: 80, 3: 160, 4: 320}

PER_BINS = [5, 20, 60, 90]
PPDU_OCCUPANCY_BINS = [0.5, 0.85, 1.0, 1.2]
OVERHEAD_RATIO_BINS = [0.06, 0.15, 0.35, 0.8]

STATE_FEATURE_NAMES = (
    "prev_per_bin",
    "prev_tail_err_flag",
    "prev_ppdu_occupancy_bin",
    "prev_action_idx",
    "cur_mcs_nss_group",
    "cur_overhead_ratio_bin",
)

COARSE_STATE_INDEX = (0, 1, 4, 5)

PREAMBLE_US = 68.0
SIFS_US = 16.0
BLOCK_ACK_US = 32.0
_BW_IN_PATH_PATTERN = re.compile(r"(?<!\d)(20|40|80|160|320)\s*[mM](?:hz)?", flags=re.IGNORECASE)


def _new_action_int_array():
    return np.zeros(len(ACTION_VALUES), dtype=np.int32)


def _new_action_float_array():
    return np.zeros(len(ACTION_VALUES), dtype=np.float64)


def load_pickle(path):
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def digitize(value, bins):
    return int(np.digitize([float(value)], bins)[0])


def to_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def infer_bw_mhz(sample, file_path):
    raw_bw = getattr(sample, "bw", None)
    if raw_bw is not None:
        try:
            raw_bw_int = int(raw_bw)
            if raw_bw_int in BW_FIELD_TO_MHZ:
                return BW_FIELD_TO_MHZ[raw_bw_int]
            if raw_bw_int in BW_RATE_NORMALIZE_FACTOR:
                return raw_bw_int
        except (TypeError, ValueError):
            pass

    matched = _BW_IN_PATH_PATTERN.search(file_path)
    if matched:
        return int(matched.group(1))
    # 当前无法识别带宽时，按 20MHz 处理，避免无谓样本丢弃。
    return 20


def mcs_nss_to_group(sample):
    """
    轻量分档策略:
    - 保留 2SS MCS6~13 为独立 8 档
    - 其余组合合并为一档(8)
    """
    mcs = int(to_float(getattr(sample, "mcs", -1), default=-1))
    nss = int(to_float(getattr(sample, "nss", 1), default=1))
    if nss >= 2 and 6 <= mcs <= 13:
        return mcs - 6  # 0~7
    return 8


def safe_action_index(real_ppdu_time_lim):
    action_value = int(real_ppdu_time_lim)
    if action_value not in ACTION_TO_INDEX:
        raise ValueError(f"Unsupported real_ppdu_time_lim: {action_value}")
    return ACTION_TO_INDEX[action_value]


def calc_throughput_reward(sample, bw_mhz, tail_penalty=0.0, reward_scale=1.0):
    """
    Convert the observed PHY rate back to a 20MHz-equivalent rate before computing reward.

    This keeps the reward definition consistent across bandwidths while avoiding
    putting bandwidth itself into the state.
    """
    payload_time_us = max(to_float(sample.tx_ppdu_time) - PREAMBLE_US, 0.0)
    airtime_us = (
        max(to_float(sample.tx_ppdu_time), to_float(sample.sw_proc_delay))
        + to_float(sample.avg_edca_delay)
        + to_float(sample.avg_rts_cts_time)
        + SIFS_US
        + BLOCK_ACK_US
    )
    bw_mhz = int(to_float(bw_mhz, 20))
    bw_factor = float(BW_RATE_NORMALIZE_FACTOR.get(bw_mhz, 1.0))
    rate_20m_equiv = to_float(sample.rate_mbps) / max(bw_factor, 1e-6)
    reward = rate_20m_equiv * payload_time_us * (1.0 - to_float(sample.per) / 100.0) / max(airtime_us, 1e-6)
    reward = reward / max(float(reward_scale), 1e-6)
    reward -= tail_penalty * int(getattr(sample, "tail_err_flag", 0))
    return float(round(reward, 6)), float(round(rate_20m_equiv, 6))


def calc_overhead_ratio(cur_sample, prev_limit_us):
    fixed_overhead = to_float(cur_sample.avg_edca_delay) + to_float(cur_sample.avg_rts_cts_time) + SIFS_US + BLOCK_ACK_US
    sw_overhang = max(to_float(cur_sample.sw_proc_delay) - float(prev_limit_us), 0.0)
    return (fixed_overhead + sw_overhang) / max(float(prev_limit_us), 1.0)


def collect_downlink_samples_by_rx_mac(samples):
    """
    Robustly regroup one raw sequence by rx_mac.

    Even if the pickle top-level key already equals rx_mac today, we regroup from
    the sample field itself so the dataset builder never creates a transition that
    crosses between different STAs.
    """
    grouped = defaultdict(list)
    for item in samples:
        if getattr(item, "direction", None) != 0 or getattr(item, "is_data", None) != 1:
            continue
        rx_mac = getattr(item, "rx_mac", None) or "unknown_rx_mac"
        grouped[rx_mac].append(item)
    return grouped


def build_state(prev_entry, cur_entry):
    """
    Build a compact state from the previous transmission result and current channel cost.

    The current STA identity is intentionally excluded from the state, but the dataset
    builder guarantees that two consecutive samples always belong to the same rx_mac.
    """
    prev_sample = prev_entry["sample"]
    cur_sample = cur_entry["sample"]
    prev_limit = max(to_float(prev_sample.real_ppdu_time_lim), 1.0)
    prev_occupancy = to_float(prev_sample.tx_ppdu_time) / prev_limit
    overhead_ratio = calc_overhead_ratio(cur_sample, prev_limit)
    state = (
        digitize(prev_sample.per, PER_BINS),
        int(prev_sample.tail_err_flag),
        digitize(prev_occupancy, PPDU_OCCUPANCY_BINS),
        safe_action_index(prev_sample.real_ppdu_time_lim),
        int(cur_entry["mcs_nss_group"]),
        digitize(overhead_ratio, OVERHEAD_RATIO_BINS),
    )
    return state


def coarse_state_from_full(state):
    return tuple(state[index] for index in COARSE_STATE_INDEX)


def _sanitize_sheet_name(candidate_name):
    sheet_name = re.sub(r"[:\\\\/*?\\[\\]]", "_", candidate_name)
    return sheet_name[:31] if len(sheet_name) > 31 else sheet_name


def _resolve_sheet_name(used_names, file_index, path):
    parent_name = os.path.basename(os.path.dirname(path)) or f"file_{file_index:02d}"
    grand_parent_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if grand_parent_name:
        candidate_label = f"{file_index:02d}_{grand_parent_name}_{parent_name}"
    else:
        candidate_label = f"{file_index:02d}_{parent_name}"
    base_name = _sanitize_sheet_name(candidate_label)
    if not base_name:
        base_name = f"sheet_{file_index:02d}"

    candidate = base_name
    suffix = 1
    while candidate in used_names:
        tail = f"_{suffix}"
        candidate = _sanitize_sheet_name(base_name[: 31 - len(tail)] + tail)
        suffix += 1
    used_names.add(candidate)
    return candidate


def export_debug_excel(
    excel_path,
    file_debug_rows,
    file_stats,
    summary,
    episode_meta,
    state_visit_counts,
    state_transition_counts,
):
    """Export a richer workbook for manual inspection of data processing and transitions."""
    if not excel_path:
        return

    try:
        import pandas as pd
    except ImportError as import_err:
        raise ImportError("pandas/openpyxl is required to export debug Excel.") from import_err

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    used_sheet_names = set()

    overview_rows = [{"metric": key, "value": value} for key, value in summary.items() if not isinstance(value, (dict, list, tuple))]

    drop_reason_rows = []
    for reason, count in sorted(summary.get("dropped_reason_counts", {}).items()):
        drop_reason_rows.append({"drop_reason": reason, "count": int(count)})

    file_summary_rows = []
    for path in sorted(file_stats):
        stats = file_stats[path]
        keep_ratio = float(stats["kept_samples"] / max(stats["raw_samples"], 1))
        file_summary_rows.append(
            {
                "file_path": path,
                "file_name": os.path.basename(os.path.dirname(path)),
                "raw_samples": int(stats["raw_samples"]),
                "kept_samples": int(stats["kept_samples"]),
                "keep_ratio": keep_ratio,
                "transitions": int(stats["transitions"]),
                "episodes": int(stats["episodes"]),
                "drop_missing_fields": int(stats["drop_reasons"].get("missing_fields", 0)),
                "drop_unsupported_action": int(stats["drop_reasons"].get("unsupported_action", 0)),
                "drop_per_too_high": int(stats["drop_reasons"].get("per_too_high", 0)),
                "drop_tx_ppdu_too_short": int(stats["drop_reasons"].get("tx_ppdu_too_short", 0)),
                "drop_low_occupancy": int(stats["drop_reasons"].get("low_occupancy", 0)),
            }
        )

    rx_mac_summary = defaultdict(lambda: {"episodes": 0, "transitions": 0, "files": set()})
    episode_summary_rows = []
    for meta in episode_meta:
        rx_mac = str(meta["rx_mac"])
        rx_mac_summary[rx_mac]["episodes"] += 1
        rx_mac_summary[rx_mac]["transitions"] += int(meta["num_transitions"])
        rx_mac_summary[rx_mac]["files"].add(meta["file_path"])
        episode_summary_rows.append(
            {
                "episode_id": int(meta["episode_id"]),
                "rx_mac": rx_mac,
                "file_path": meta["file_path"],
                "file_name": os.path.basename(os.path.dirname(meta["file_path"])),
                "seq_key": meta["seq_key"],
                "start_transition_idx": int(meta["start_transition_idx"]),
                "end_transition_idx": int(meta["end_transition_idx"]),
                "num_transitions": int(meta["num_transitions"]),
            }
        )

    rx_mac_summary_rows = []
    for rx_mac, stats in sorted(rx_mac_summary.items()):
        rx_mac_summary_rows.append(
            {
                "rx_mac": rx_mac,
                "episodes": int(stats["episodes"]),
                "transitions": int(stats["transitions"]),
                "num_files": int(len(stats["files"])),
            }
        )

    top_state_rows = []
    for state, count in sorted(state_visit_counts.items(), key=lambda item: item[1], reverse=True)[:200]:
        top_state_rows.append({"state": str(state), "visit_count": int(count)})

    top_transition_rows = []
    for (state, next_state), count in sorted(state_transition_counts.items(), key=lambda item: item[1], reverse=True)[:500]:
        top_transition_rows.append(
            {
                "state": str(state),
                "next_state": str(next_state),
                "transition_count": int(count),
            }
        )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(overview_rows).to_excel(writer, sheet_name="overview", index=False)
        pd.DataFrame(drop_reason_rows).to_excel(writer, sheet_name="drop_reasons", index=False)
        pd.DataFrame(file_summary_rows).to_excel(writer, sheet_name="file_summary", index=False)
        pd.DataFrame(episode_summary_rows).to_excel(writer, sheet_name="episode_summary", index=False)
        pd.DataFrame(rx_mac_summary_rows).to_excel(writer, sheet_name="rx_mac_summary", index=False)
        pd.DataFrame(top_state_rows).to_excel(writer, sheet_name="top_states", index=False)
        pd.DataFrame(top_transition_rows).to_excel(writer, sheet_name="top_transitions", index=False)

        for file_index, path in enumerate(sorted(file_stats), start=1):
            records = file_debug_rows.get(path, [])
            sheet_name = _resolve_sheet_name(used_sheet_names, file_index, path)
            if records:
                df = pd.DataFrame.from_records(records)
            else:
                df = pd.DataFrame([{"info": "No transitions survived filters for this file.", "file_path": path}])
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def validate_sample(sample, per_high_cutoff, min_tx_ppdu_time_us, min_occupancy_for_update):
    """Apply sample-level filtering before any state transition is formed."""
    required_fields = ("per", "tx_ppdu_time", "real_ppdu_time_lim", "sw_proc_delay", "avg_edca_delay", "avg_rts_cts_time")
    for field_name in required_fields:
        if getattr(sample, field_name, None) is None:
            return False, "missing_fields"

    try:
        _ = safe_action_index(sample.real_ppdu_time_lim)
    except ValueError:
        return False, "unsupported_action"

    if to_float(sample.per) >= float(per_high_cutoff):
        return False, "per_too_high"

    tx_ppdu_time = to_float(sample.tx_ppdu_time)
    if tx_ppdu_time < float(min_tx_ppdu_time_us):
        return False, "tx_ppdu_too_short"

    if float(min_occupancy_for_update) > 0:
        occupancy = tx_ppdu_time / max(to_float(sample.real_ppdu_time_lim), 1.0)
        if occupancy < float(min_occupancy_for_update):
            return False, "low_occupancy"

    return True, "ok"


def build_offline_dataset(
    data_root=RAW_DATA_ROOT,
    output_path=DEFAULT_DATASET_PATH,
    tail_penalty=0.0,
    per_high_cutoff=90.0,
    min_tx_ppdu_time_us=600.0,
    min_occupancy_for_update=0.0,
    reward_scale=1.0,
    debug_excel_path=None,
):
    """
    Build an offline dataset for Q-learning.

    Key rules:
    - transitions are only formed inside the same rx_mac stream
    - one rx_mac stream corresponds to one episode
    - filtered samples are skipped, but they do not split the episode
    - the last transition of the rx_mac stream is marked done=True
    """
    dfx_paths = sorted(glob.glob(os.path.join(data_root, "**", "dfx_data.pkl"), recursive=True))
    if not dfx_paths:
        raise FileNotFoundError(f"No dfx_data.pkl found under: {data_root}")

    transitions = []
    episode_slices = []
    episode_meta = []
    rx_mac_episode_groups = defaultdict(list)
    state_action_counts = defaultdict(_new_action_int_array)
    state_action_reward_sum = defaultdict(_new_action_float_array)
    coarse_state_action_counts = defaultdict(_new_action_int_array)
    coarse_state_action_reward_sum = defaultdict(_new_action_float_array)
    global_action_counts = _new_action_int_array()
    global_action_reward_sum = _new_action_float_array()
    state_visit_counts = Counter()
    state_transition_counts = Counter()

    file_debug_rows = defaultdict(list)
    file_stats = {}
    dropped_reason_counts = Counter()

    num_raw_sequences = 0
    num_valid_segments = 0
    skipped_short_sequences = 0
    skipped_short_segments_after_filter = 0
    num_raw_samples = 0
    num_kept_samples = 0
    reward_values = []
    num_rx_streams = 0

    for path in dfx_paths:
        file_stats[path] = {
            "raw_samples": 0,
            "kept_samples": 0,
            "transitions": 0,
            "episodes": 0,
            "drop_reasons": Counter(),
        }
        loaded = load_pickle(path)
        for seq_key, samples in loaded.items():
            grouped_samples = collect_downlink_samples_by_rx_mac(samples)
            if not grouped_samples:
                skipped_short_sequences += 1
                continue

            num_raw_sequences += 1
            for rx_mac, downlink_data in grouped_samples.items():
                if len(downlink_data) < 3:
                    skipped_short_sequences += 1
                    continue

                num_rx_streams += 1
                valid_entries = []
                for item in downlink_data:
                    num_raw_samples += 1
                    file_stats[path]["raw_samples"] += 1

                    bw_mhz = infer_bw_mhz(item, path)
                    is_valid, reason = validate_sample(
                        item,
                        per_high_cutoff=per_high_cutoff,
                        min_tx_ppdu_time_us=min_tx_ppdu_time_us,
                        min_occupancy_for_update=min_occupancy_for_update,
                    )
                    if not is_valid:
                        dropped_reason_counts[reason] += 1
                        file_stats[path]["drop_reasons"][reason] += 1
                        continue

                    entry = {
                        "sample": item,
                        "bw_mhz": int(bw_mhz),
                        "mcs_nss_group": int(mcs_nss_to_group(item)),
                        "rx_mac": str(rx_mac),
                    }
                    valid_entries.append(entry)
                    num_kept_samples += 1
                    file_stats[path]["kept_samples"] += 1

                if len(valid_entries) < 3:
                    skipped_short_segments_after_filter += 1
                    continue

                num_valid_segments += 1
                episode_id = len(episode_slices)
                episode_start = len(transitions)
                for index in range(1, len(valid_entries) - 1):
                    prev_entry = valid_entries[index - 1]
                    cur_entry = valid_entries[index]
                    next_entry = valid_entries[index + 1]

                    prev_sample = prev_entry["sample"]
                    cur_sample = cur_entry["sample"]
                    next_sample = next_entry["sample"]

                    state = build_state(prev_entry, cur_entry)
                    next_state = build_state(cur_entry, next_entry)
                    action = safe_action_index(cur_sample.real_ppdu_time_lim)
                    reward, rate_20m_equiv = calc_throughput_reward(
                        cur_sample,
                        bw_mhz=cur_entry["bw_mhz"],
                        tail_penalty=tail_penalty,
                        reward_scale=reward_scale,
                    )
                    done = index == len(valid_entries) - 2

                    transitions.append((state, action, reward, next_state, done))
                    file_stats[path]["transitions"] += 1

                    state_action_counts[state][action] += 1
                    state_action_reward_sum[state][action] += reward

                    coarse_state = coarse_state_from_full(state)
                    coarse_state_action_counts[coarse_state][action] += 1
                    coarse_state_action_reward_sum[coarse_state][action] += reward

                    global_action_counts[action] += 1
                    global_action_reward_sum[action] += reward
                    reward_values.append(reward)
                    state_visit_counts[state] += 1
                    state_transition_counts[(state, next_state)] += 1

                    prev_limit = max(to_float(prev_sample.real_ppdu_time_lim), 1.0)
                    file_debug_rows[path].append(
                        {
                            "episode_id": int(episode_id),
                            "seq_key": str(seq_key),
                            "rx_mac": str(rx_mac),
                            "episode_transition_idx": int(index - 1),
                            "episode_num_transitions": int(len(valid_entries) - 2),
                            "transition_idx": index,
                            "bw_mhz": int(cur_entry["bw_mhz"]),
                            "mcs_nss_group": int(cur_entry["mcs_nss_group"]),
                            "action_idx": int(action),
                            "action_value": int(ACTION_VALUES[action]),
                            "reward_train": float(reward),
                            "rate_20m_equiv_mbps": float(rate_20m_equiv),
                            "state": str(state),
                            "next_state": str(next_state),
                            "prev_mcs": int(to_float(prev_sample.mcs)),
                            "prev_nss": int(to_float(prev_sample.nss)),
                            "prev_rate_mbps": float(to_float(prev_sample.rate_mbps)),
                            "prev_per": float(to_float(prev_sample.per)),
                            "prev_tail_err_flag": int(getattr(prev_sample, "tail_err_flag", 0)),
                            "prev_ppdu_occupancy": float(round(to_float(prev_sample.tx_ppdu_time) / prev_limit, 6)),
                            "prev_tx_ppdu_time": float(to_float(prev_sample.tx_ppdu_time)),
                            "prev_sw_proc_delay": float(to_float(prev_sample.sw_proc_delay)),
                            "prev_avg_edca_delay": float(to_float(prev_sample.avg_edca_delay)),
                            "prev_avg_rts_cts_time": float(to_float(prev_sample.avg_rts_cts_time)),
                            "prev_action_value": int(to_float(prev_sample.real_ppdu_time_lim)),
                            "cur_per": float(to_float(cur_sample.per)),
                            "cur_mcs": int(to_float(cur_sample.mcs)),
                            "cur_nss": int(to_float(cur_sample.nss)),
                            "cur_rate_mbps": float(to_float(cur_sample.rate_mbps)),
                            "cur_sw_proc_delay": float(to_float(cur_sample.sw_proc_delay)),
                            "cur_avg_edca_delay": float(to_float(cur_sample.avg_edca_delay)),
                            "cur_avg_rts_cts_time": float(to_float(cur_sample.avg_rts_cts_time)),
                            "cur_tx_ppdu_time": float(to_float(cur_sample.tx_ppdu_time)),
                            "cur_real_ppdu_time_lim": int(to_float(cur_sample.real_ppdu_time_lim)),
                            "cur_overhead_ratio": float(round(calc_overhead_ratio(cur_sample, prev_limit), 6)),
                            "next_per": float(to_float(next_sample.per)),
                            "next_mcs": int(to_float(next_sample.mcs)),
                            "next_nss": int(to_float(next_sample.nss)),
                            "next_rate_mbps": float(to_float(next_sample.rate_mbps)),
                            "next_tx_ppdu_time": float(to_float(next_sample.tx_ppdu_time)),
                            "next_real_ppdu_time_lim": int(to_float(next_sample.real_ppdu_time_lim)),
                            "done": bool(done),
                            "is_episode_last": bool(done),
                        }
                    )

                episode_end = len(transitions)
                if episode_end > episode_start:
                    episode_slices.append((episode_start, episode_end))
                    episode_meta.append(
                        {
                            "episode_id": episode_id,
                            "file_path": path,
                            "seq_key": str(seq_key),
                            "rx_mac": str(rx_mac),
                            "start_transition_idx": int(episode_start),
                            "end_transition_idx": int(episode_end - 1),
                            "num_transitions": int(episode_end - episode_start),
                        }
                    )
                    rx_mac_episode_groups[str(rx_mac)].append(episode_id)
                    file_stats[path]["episodes"] += 1

    if not transitions:
        raise RuntimeError("No valid transitions were built from the raw dataset.")

    global_action_mean_reward = np.divide(
        global_action_reward_sum,
        np.maximum(global_action_counts, 1),
        dtype=np.float64,
    )
    default_action_idx = int(np.argmax(global_action_mean_reward))

    action_coverages = [int(np.count_nonzero(action_counts)) for action_counts in state_action_counts.values()]
    summary = {
        "data_root": os.path.abspath(data_root),
        "num_dfx_files": len(dfx_paths),
        "num_raw_sequences": num_raw_sequences,
        "num_rx_streams": num_rx_streams,
        "num_episodes": len(episode_slices),
        "num_valid_segments": num_valid_segments,
        "skipped_short_sequences": skipped_short_sequences,
        "skipped_short_segments_after_filter": skipped_short_segments_after_filter,
        "num_raw_samples": int(num_raw_samples),
        "num_kept_samples": int(num_kept_samples),
        "sample_keep_ratio": float(num_kept_samples / max(num_raw_samples, 1)),
        "num_transitions": len(transitions),
        "num_states": len(state_action_counts),
        "num_coarse_states": len(coarse_state_action_counts),
        "avg_actions_per_state": float(np.mean(action_coverages)),
        "behavior_reward_mean": float(np.mean(reward_values)),
        "default_action_idx": default_action_idx,
        "default_action_value": int(ACTION_VALUES[default_action_idx]),
        "tail_penalty": float(tail_penalty),
        "per_high_cutoff": float(per_high_cutoff),
        "min_tx_ppdu_time_us": float(min_tx_ppdu_time_us),
        "min_occupancy_for_update": float(min_occupancy_for_update),
        "reward_scale": float(reward_scale),
        "bw_rate_normalize_factor": dict(BW_RATE_NORMALIZE_FACTOR),
        "dropped_reason_counts": dict(dropped_reason_counts),
        "debug_excel_path": os.path.abspath(debug_excel_path) if debug_excel_path else None,
    }

    dataset = {
        "transitions": transitions,
        "action_values": ACTION_VALUES,
        "feature_names": STATE_FEATURE_NAMES,
        "feature_bins": {
            "per_bins": PER_BINS,
            "ppdu_occupancy_bins": PPDU_OCCUPANCY_BINS,
            "overhead_ratio_bins": OVERHEAD_RATIO_BINS,
            "mcs_nss_group_desc": {
                "0-7": "2SS MCS6~13",
                "8": "Others",
            },
        },
        "state_action_counts": dict(state_action_counts),
        "state_action_reward_sum": dict(state_action_reward_sum),
        "coarse_state_action_counts": dict(coarse_state_action_counts),
        "coarse_state_action_reward_sum": dict(coarse_state_action_reward_sum),
        "state_visit_counts": dict(state_visit_counts),
        "state_transition_counts": dict(state_transition_counts),
        "global_action_counts": global_action_counts,
        "global_action_reward_sum": global_action_reward_sum,
        "episode_slices": episode_slices,
        "episode_meta": episode_meta,
        "rx_mac_episode_groups": dict(rx_mac_episode_groups),
        "file_stats": {
            path: {
                **stats,
                "drop_reasons": dict(stats["drop_reasons"]),
            }
            for path, stats in file_stats.items()
        },
        "summary": summary,
    }

    export_debug_excel(
        debug_excel_path,
        file_debug_rows=file_debug_rows,
        file_stats=file_stats,
        summary=summary,
        episode_meta=episode_meta,
        state_visit_counts=state_visit_counts,
        state_transition_counts=state_transition_counts,
    )

    if output_path:
        save_pickle(dataset, output_path)

    return dataset
