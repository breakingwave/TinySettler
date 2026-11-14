"""CPU-only resource accounting and descriptive training-return curves."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def main(out):
    read = lambda name: json.loads((out / name).read_text())
    controller = read('controller.json')
    attempts = controller['previous_attempts'] + [controller]
    ledger = []
    for allocation, attachment, attempt in zip(controller['allocations'], controller['attachments'], attempts):
        assert allocation['session'] == attempt['session']
        ledger.append(dict(session=allocation['session'], request_started=attachment['time'],
                           ready=allocation['allocated'], stopped=attempt['ended'],
                           allocation_upper_bound_seconds=attempt['ended'] - attachment['time'],
                           termination_verified=attempt['termination_verified']))
    runs = read('runs.json')
    episodes = list(csv.DictReader((out / 'training-episodes.csv').open()))
    resources = dict(allocations=ledger,
                     total_allocation_upper_bound_seconds=sum(x['allocation_upper_bound_seconds'] for x in ledger),
                     conservative_original_clock_seconds=controller['total_allocated_upper_bound_seconds'],
                     original_deadline=controller['deadline'],
                     retained_training_seconds=sum(x['stats']['elapsed'] for x in runs),
                     note='Retained training time excludes rolled-back work; allocation intervals include setup, evaluation, transfer and lost work.')
    resources['training_by_arm'] = {}
    for arm in ['standard', 'curriculum']:
        rs = [r for r in runs if r['arm'] == arm]
        es = [e for e in episodes if e['arm'] == arm]
        categories = Counter()
        planning = Counter()
        for r in rs:
            categories.update(r['category_transitions'])
            planning.update(r['planning_totals'])
        n = sum(categories.values())
        terminals = sum(e['terminal'] == 'True' for e in es)
        resources['training_by_arm'][arm] = dict(transitions=n, category_transitions=dict(categories),
            completed_episodes=len(es), terminal_episodes=terminals, truncations=len(es)-terminals,
            terminal_reward_frequency_per_transition=terminals/n,
            realized_episode_opponents=dict(Counter(e['opponent'] for e in es)), planning=dict(planning))
    gpu = []
    cpu = []
    for p in out.glob('telemetry*.csv'):
        gpu.extend([[float(v.strip()) for v in r['gpu_csv'].split(',')[1:]] for r in csv.DictReader(p.open())])
    for p in out.glob('cpu-telemetry*.csv'):
        cpu.extend(csv.DictReader(p.open()))
    resources['telemetry'] = dict(samples=len(gpu), mean_gpu_utilization_pct=sum(x[0] for x in gpu)/len(gpu),
        peak_gpu_memory_mib=max(x[2] for x in gpu),
        mean_trainer_cpu_pct_one_core=sum(float(x['trainer_cpu_pct_one_core']) for x in cpu)/len(cpu),
        peak_trainer_rss_gib=max(float(x['trainer_rss_bytes']) for x in cpu)/2**30,
        scope='All recorded training/evaluation samples across all three sequential allocations; two CPU cores.')
    (out / 'resource-summary.json').write_text(json.dumps(resources, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/tinysettler-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    rows = []
    for ax, arm in zip(axes, ['standard', 'curriculum']):
        for seed in [104729, 130363, 155921]:
            es = [e for e in episodes if e['arm'] == arm and int(e['seed']) == seed and e['category'] == 'fresh']
            xs, ys = [], []
            for i in range(99, len(es), 20):
                window = es[i-99:i+1]
                n = int(es[i]['n'])
                reward = sum(float(e['reward']) for e in window)/100
                xs.append(n); ys.append(reward)
                rows.append(dict(arm=arm, seed=seed, n=n, fresh_episode_window=100, mean_terminal_reward=reward))
            ax.plot(xs, ys, label=str(seed))
        ax.set(title=arm, xlabel='Additional learner transitions', ylabel='Mean reward: last 100 fresh training episodes', ylim=(-1, 1))
        ax.legend()
    fig.suptitle('Descriptive training returns; not held-out performance')
    fig.tight_layout(); fig.savefig(out/'training-return-curves.png', dpi=160); plt.close(fig)
    with (out/'training-return-curves.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/'postprocessing-source-hashes.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name('curriculum_report.py'), Path(__file__).with_name('verify_results.py')]}, indent=2))
    print(json.dumps(resources, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
