import subprocess
import sys
from pathlib import Path
import yaml
import json
import csv

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
    else:
        print(f"Warning: results file not found at {res_path}")
    tmp_cfg.unlink(missing_ok=True)
    return result

def aggregate_results(results, csv_path=OUTPUT_CSV):
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
    results = [run_experiment(e) for e in experiments]
    csv_path = aggregate_results(results)
    print(f"Aggregated results saved to {csv_path}")

if __name__ == '__main__':
    main()
