# Scaling study: preflight failed; no controlled comparison

The implementation and bounded CPU checks were completed, and two dedicated Colab T4 allocations were attempted. **Calibration failed before scope, transition budget or evaluation coverage could be frozen. None of the 16 experimental runs started.** There are no final strength results, selected deployment policy, or valid algorithm rankings from this attempt.

The attempt is closed after its one permitted fault replacement. Both dedicated runtimes were stopped and independently confirmed absent with `colab sessions`. Request-to-stop allocation totaled **78.27 minutes**; the original shared clock elapsed **83.76 minutes**, below the 840-minute hard limit. The original deadline was never reset. Model monitoring consisted of the first hourly review and fatal-failure interventions; deterministic infrastructure performed ordinary checks and transfers.

## What happened

1. The first T4 ran fresh single/dual-worker A–D pilots, each with 8,192 transitions per worker, and passed their exact CUDA continuation checks. E completed 8,192 transitions but its subsequent exact recovery check diverged. The supervisor retrieved diagnostics and terminated the runtime.
2. Inspection found a real RNG isolation defect: CPU model construction called `torch.manual_seed` within a CPU-only `fork_rng`. That call also reseeded CUDA, so an opponent cache miss could change the learner's minibatch RNG. The repair uses `torch.random.default_generator.manual_seed` for CPU initialization. Regression checks were added; the replacement passed an explicit actual-CUDA state-invariance assertion. **The complete graph CUDA recovery test was not reached again, so the repair is not claimed to close every possible cause of that graph failure.**
3. A separately hashed repair bundle preserved the failed bundle and original clock. The single allowed fault replacement reran pilots from scratch. Up to 30 minutes of the existing recovery reserve was explicitly made available to preflight; no additional total time was granted. Dependency setup and CLI timeouts consumed much of that allowance. A–D single/dual pilots and recovery checks passed again. E stopped at **6,719/8,192 transitions** when the combined preflight allowance expired. The manager rejected the incomplete pilot. Subsequent bounded CLI retries and final retrieval delayed termination until minute 83.76; this was retrieval/cleanup, not continued experimental training.
4. F/G pilots, fresh evaluation timing, external checkpoint upload/resume verification, scope freezing and the experimental sweep were never reached. No further fault allocation was made.

These are preflight/infrastructure failures, not evidence of PPO regression or failure to learn. Pilot metrics must not be treated as experimental learning curves or used to select models. Across both attempts, pilots collected 211,519 transitions, including the incomplete E pilot; **retained experimental transitions are zero**.

## Implemented and checked

The new [implementation](../../scaling/README.md) provides a shared 879-feature deterministic scaling schema, MLP-128/512 and a 932,098-parameter graph/candidate model, immutable actor-relative neural opponents, static/refreshed pools, a bounded teacher corpus, instrumented masked PPO, paired scheduling, calibration-based scope reduction, independent evaluation, atomic full-state recovery, verified external generations, a deadline watchdog and reporting tools. Historical artifacts and the browser demo were preserved.

Fresh training seeds are 400009, 400031 and 400033. Source-game namespaces and teacher splits were recorded before execution. The teacher corpus contains 200 legal source games and 9,635/1,334/1,301 stored train/validation/test decisions, split by source game. Six historical policies were exported only after provenance and prediction checks; a seventh policy from lineage 196613 was reserved for evaluation. See [teacher provenance](teacher-provenance.json), [archive provenance](archive-provenance.json), and the [full input manifest](raw/repair1/input-manifest.json).

Local checks covered inherited privacy/legality/reward/turn contracts; schema and candidate mapping; A/B/E/G collection, optimization and CPU recovery; physical sampled-world conservation; frozen opponents; configured mixture frequencies; paired initialization; independent-process recovery; and evaluation resume. The final repository has **10 passing controller/isolation tests**. Two post-run controller safeguards prevent fatal `--resume` from bypassing recovery limits and clear stale `all_terminated` status during allocation. These safeguards were tested locally and were **not part of the executed bundles**; their hashes are identified separately. No new runtime was allocated to test them.

The real device was Tesla T4, with Torch 2.9.1+cu128 and two exposed CPU cores. A real CUDA Adam update passed on both allocations. Search was omitted before allocation because a repaired search opponent had not passed its required correctness gate. The [dependency lock](../../scaling/requirements-lock.txt) and archived installed-package inventories are retained.

## Supported measurements

Fresh end-to-end pilot rates include worker startup and checkpoint serialization; two-worker rates are aggregate decisions/s on the same T4. They do not include the subsequent recovery probe. These are short timing samples, not performance estimates for a complete training trajectory.

| Condition | Initial: one / two workers | Repair: one / two workers | Experimental result |
|---|---:|---:|---|
| A: MLP-128 static | 177.9 / 199.7 | 163.7 / 178.0 | Not started |
| B: MLP-128 refreshed | 186.1 / 194.9 | 186.1 / 186.1 | Not started |
| C: MLP-512 static | 178.0 / 194.9 | 163.7 / 174.2 | Not started |
| D: MLP-512 refreshed | 186.1 / 186.1 | 151.6 / 174.2 | Not started |
| E: graph static | Recovery failed | Pilot incomplete | Not started |
| F: graph refreshed | Not reached | Not reached | Not started |
| G: teacher + refreshed MLP-128 | Not reached | Not reached | Not started |

Two processes did not provide a consistent gain above the predeclared 2% threshold across conditions; doubling throughput is not supported. The first graph worker measured 61.2 learner decisions/s inside its training loop, versus roughly 200–218 for the single-worker MLP pilots, but failed recovery. That is provisional cost evidence only, and is not directly interchangeable with the end-to-end rates above.

[Worker metrics](pilot-workers.csv) include engine actions/s and update time. [Timing records](pilot-timings.csv), [telemetry](telemetry.csv), [all 16 missing comparison cells](comparison-status.csv) and raw per-worker records are available. PNG and SVG charts show [pilot throughput](charts/01-pilot-throughput.png), [allocation timeline](charts/02-allocation-timeline.png) and [CPU/GPU utilization](charts/03-preflight-utilization.png). Their labels were visually inspected.

The requested strength curves, heatmaps, paired uncertainty intervals, strength/cost frontier, experimental action-use/critic/pool curves and sampling-versus-greedy comparison cannot be computed: no experimental validation or test games exist. [Chart availability](chart-availability.json) makes these omissions explicit. No blank strength plot or invented interval substitutes for missing evidence.

## Ranked next direction

This is a planning recommendation, **not a ranking of learned policies**:

1. **Complete CUDA recovery and remote setup validation, then prioritize the three-pair A/B comparison.** It directly tests the primary opponent-refresh question at the lowest measured model cost. Before another allocation, make installation asynchronous with a bounded stage deadline and preserve a tiny failing checkpoint externally; then verify the repaired graph continuation across cache misses. Strongest counterevidence: this attempt supplies no external-strength result showing refreshed self-play helps, and early self snapshots may be weaker than historical opponents.
2. **Add C/D only when the complete conservative timing projection fits.** Short MLP-512 pilots were close to MLP-128 cost, so capacity may be affordable. Counterevidence: five times as many parameters have no demonstrated strength benefit here; pilot throughput is not later-stage throughput.
3. **Defer the graph and teacher comparisons until their full gates pass.** Graph cost and unresolved end-to-end CUDA recovery create a substantial execution risk. Teacher data is prepared, but its optimization/evaluation cost and downstream benefit were not calibrated. Neither comparison can be ranked on playing strength.

Refreshing opponents, more parameters, graph structure, teacher initialization, seed reliability, long-training benefit and transfer of sampling's earlier diagnostic advantage **all remain unanswered**. A larger GPU is not justified by this attempt. This study does not measure negotiation quality or browser latency, and PPO probabilities are not calibrated action values.

## Artifacts, hashes and reproduction

Persistent roots, outside the synced checkout:

- `/root/tinysettler-artifacts/tinysettler-scaling-20261114-a/`: original source, inputs, logs and two verified external generations.
- `/root/tinysettler-artifacts/tinysettler-scaling-20261114-a-repair1/`: repair source, inputs, logs, runtime ledger and two verified external generations.

Source archive SHA256 values:

- Initial: `e3b0e6142294dd3c00fe5c3433778bb3e5e6e528417e1574ef9684b7f8e75b97`
- Repair: `0816def1c5eda0895563c8f10cfd87ebbafdc87c5956380ca7849ad0e05f9443`

All 23 executed source hashes per bundle, both input archives, teacher/archive payloads, every object referenced by all four retained generations, and all latest retrieved files were reverified. The latest generations contain 95 and 94 files. See [verification receipts](artifact-verification.json), [source hashes](source-hashes.json), [resource accounting](resource-summary.json) and [runtime ledger](runtime-ledger.csv).

**No experimental checkpoints exist.** Pilot weight files were deliberately excluded from external generations; they are unavailable after runtime termination. This limits forensic replay of the original failure and is a recovery-design weakness to fix before another run. The immutable historical inference archive, its original checkpoint hashes, teacher data and both source bundles remain available. An end-to-end external checkpoint upload/resume round trip was not completed in this attempt.

Rebuild this failure report's data and charts without allocating a runtime:

```sh
MPLCONFIGDIR=/tmp/tinysettler-matplotlib /tmp/tinysettler-rl-venv/bin/python \
  rl_research/scaling/report_preflight_failure.py \
  --first /root/tinysettler-artifacts/tinysettler-scaling-20261114-a \
  --repair /root/tinysettler-artifacts/tinysettler-scaling-20261114-a-repair1 \
  --output rl_research/results/scaling-20261114
```

Preparation, launch and evaluation command forms are documented in the implementation README. **Do not resume these closed allocations or reset their deadline.** The repository controller rejects fatal resumes; the immutable archived controller predates that added guard. Another study requires a new explicit allocation authorization and fresh manifest. This delivery records a bounded failed attempt, not completion of the requested scientific comparison.
