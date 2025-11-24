import subprocess
import sys
from pathlib import Path
import yaml
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
MAIN_PY = ROOT / 'main.py'
CONFIG_FILE = ROOT / 'config.yaml'
REPO_ROOT = ROOT.parent.parent
OUTPUT_CSV = REPO_ROOT / 'outputs' / 'archs_summary.csv'

def load_configs(path):
    with open(path) as f:
        return yaml.safe_load(f).get('experiments', [])

def run_experiment(entry):
    tmp_cfg = REPO_ROOT / 'tmp_run_config.yaml'
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(entry, f)
    cmd = [sys.executable, str(MAIN_PY), '--config', tmp_cfg.name]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print('Running:', ' '.join(cmd))
    print('stdout:\n', proc.stdout)
    print('stderr:\n', proc.stderr)
    out_dir = entry.get('output_dir', 'outputs')
    res_path = REPO_ROOT / out_dir / 'arch_logs' / f"{entry.get('name', 'font_ae')}_results.json"
    result = None
    if res_path.exists():
        with open(res_path) as f:
            result = json.load(f)
            result['output_dir'] = out_dir
    else:
        print(f"Warning: results file not found at {res_path}")
    tmp_cfg.unlink(missing_ok=True)
    return result

def plot_loss_comparison_with_error(all_results, output_path):
    plt.figure(figsize=(12, 8))
    for name, experiment_data in all_results.items():
        runs = experiment_data['runs']
        if not runs:
            continue
        loss_histories = [run['loss_history'] for run in runs if 'loss_history' in run and run['loss_history']]
        if not loss_histories:
            print(f"No loss history found for experiment {name}")
            continue
        max_len = max(len(h) for h in loss_histories)
        padded_histories = []
        for h in loss_histories:
            if len(h) < max_len:
                # Pad with the last value
                padding = [h[-1]] * (max_len - len(h))
                padded_histories.append(h + padding)
            else:
                padded_histories.append(h)

        loss_histories_np = np.array(padded_histories)
        mean_loss = np.mean(loss_histories_np, axis=0)
        std_loss = np.std(loss_histories_np, axis=0)
        epochs = np.arange(len(mean_loss))
        line, = plt.plot(epochs, mean_loss, label=name)
        plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color=line.get_color())

    plt.title('Training Loss Comparison with Error Bands (5 runs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(output_path)
    plt.close()

def aggregate_results(results, csv_path=OUTPUT_CSV):
    if not results:
        print("No results to aggregate.")
        return

    rows = [
        {
            'name': r.get('name'),
            'topology': '-'.join(map(str, r.get('topology', []))),
            'final_loss': r.get('final_loss'),
            'pixel_errors_mean': r.get('pixel_errors_mean'),
            'pixel_errors_std': r.get('pixel_errors_std')
        }
        for r in results if r
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return csv_path

def main():
    experiments = load_configs(CONFIG_FILE)
    all_results = {}
    num_runs = 10

    for experiment_config in experiments:
        experiment_name = experiment_config['name']
        print(f"--- Running experiment: {experiment_name} ({num_runs} times) ---")

        all_results[experiment_name] = {
            'runs': [],
            'config': experiment_config
        }

        for i in range(num_runs):
            print(f"  - Run {i+1}/{num_runs}")
            run_config = experiment_config.copy()
            run_name = f"{experiment_name}_run_{i+1}"
            run_config['name'] = run_name

            original_output_dir = Path(run_config.get('output_dir', 'outputs'))
            run_output_dir = original_output_dir / f"run_{i+1}"
            run_config['output_dir'] = str(run_output_dir)

            result = run_experiment(run_config)
            if result:
                all_results[experiment_name]['runs'].append(result)

    # Flatten results for CSV aggregation
    flat_results = [run for exp_data in all_results.values() for run in exp_data['runs']]
    if flat_results:
        csv_path = aggregate_results(flat_results)
        print(f"Aggregated results saved to {csv_path}")
    else:
        print("No results to aggregate.")
    plot_output_path = REPO_ROOT / 'outputs' / 'loss_comparison_with_error.png'
    plot_loss_comparison_with_error(all_results, plot_output_path)
    print(f"Loss comparison plot with error saved to {plot_output_path}")


if __name__ == '__main__':
    main()
