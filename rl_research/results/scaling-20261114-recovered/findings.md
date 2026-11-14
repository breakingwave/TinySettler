# Recovered scaling study: completed, no established strength winner

**Recovery succeeded. All ten retained runs completed 425,984 learner transitions, and all frozen evaluation panels finished.** The controlled comparison covers MLP-128 and MLP-512 under static and refreshed opponent pools. Graph and teacher conditions passed preflight but were omitted by the predeclared throughput-based scope rule before experimental outcomes.

The main conclusion is negative: **this budget did not establish a learner that scales to strong play against the value heuristics.** Refreshed self-play did not show a reliable advantage. Static MLP-512 has a promising but inconclusive capacity signal, based on only two paired training seeds. Most losses occur before substantial expansion, rather than through administrative timeouts.

All four dedicated runtime allocations—including the two earlier failed preflights and the planned handoff—are verified stopped. Total allocated request-to-stop time was **10.98 hours**. The original shared clock elapsed **11.25 hours**, below the unchanged 14-hour limit. No learning settings were tuned during hourly monitoring.

## Primary fixed-budget results

Primary inference is masked categorical sampling at temperature 1. Each policy played 20 untouched boards, both seats, against each opponent. Entries below show wins/games; **all remaining games are losses, with zero truncations and zero illegal actions**. Base and contender are configurations of the same heuristic family. The held-out neural opponent comes from a different historical training lineage.

| Arm | Training seeds | Base | Contender | Equal-weight heuristic family | Held-out PPO | Mean training minutes/run |
|---|---:|---:|---:|---:|---:|---:|
| A: MLP-128 static | 3 | 4/120 (3.33%) | 1/120 (0.83%) | 5/240 (2.08%) | 64/120 (53.33%) | 41.57 |
| B: MLP-128 refreshed | 3 | 3/120 (2.50%) | 4/120 (3.33%) | 7/240 (2.92%) | 70/120 (58.33%) | 40.79 |
| C: MLP-512 static | 2 | 2/80 (2.50%) | 5/80 (6.25%) | 7/160 (4.38%) | 50/80 (62.50%) | 40.92 |
| D: MLP-512 refreshed | 2 | 1/80 (1.25%) | 4/80 (5.00%) | 5/160 (3.12%) | 45/80 (56.25%) | 40.89 |

[Full pooled results](pooled.csv), [per-seed/opponent results](per-seed.csv), [comparison table](comparison-table.csv), and [raw final games](final-games.csv) include losses, points, inference costs and both inference modes. The pooled table uses all retained seeds; capacity contrasts below use only the first two common seeds, so they cannot be reconstructed by simply subtracting its full three-seed A/B averages.

Per-seed primary heuristic-family wins, each out of 80 games:

| Seed | A | B | C | D |
|---|---:|---:|---:|---:|
| 400009 | 0 | 3 | 4 | 0 |
| 400031 | 1 | 0 | 3 | 5 |
| 400033 | 4 | 4 | Not scheduled | Not scheduled |

## Paired effects and uncertainty

Intervals resample source boards while retaining both seats and opponent panels. The crossed intervals additionally resample paired training seeds. They are exploratory bootstrap estimates: three primary seed pairs and two capacity pairs do not characterize training variability precisely.

| Contrast | Heuristic-family difference, percentage points | Board-only 95% interval | Crossed seed/board 95% interval |
|---|---:|---:|---:|
| B − A: refreshed self-play, MLP-128 | +0.83 | [−2.08, +4.17] | **[−3.75, +5.83]** |
| C − A: capacity under static opponents | +3.75 | [0.00, +8.13] | **[−1.25, +8.75]** |
| D − B: capacity under refreshed opponents | +1.25 | [−1.25, +4.38] | **[−6.25, +10.00]** |
| (D − C) − (B − A): capacity × refresh interaction | −2.50 | [−8.13, +3.13] | **[−12.50, +8.75]** |

B−A's individual heuristic effects were +3.75, −1.25 and 0 points. Its held-out PPO effect was +5.0 points, crossed interval [−13.33, +27.50], driven by one seed's +22.5-point improvement while the other two declined. This does not establish a repeatable self-play gain.

Static capacity improved the heuristic endpoint by +5.0 and +2.5 points in its two pairs. Its held-out PPO effect was +17.5 points: the board-only interval [2.5, 32.5] excludes zero, but the crossed interval **[−7.5, +42.5]** does not. The narrow board-only result must not be presented as reliable architecture evidence. Refreshed capacity and factorial interactions are also inconclusive. Graph structure and teacher initialization remain **unanswered**, not equivalent or inferior. See [paired analyses](paired-comparisons.json).

## Sampling, selection and confirmation

The earlier sampling advantage in selected failure positions did **not** transfer as a universal ordinary-game advantage. Against the heuristic family, sampled/greedy win rates were A 2.08%/6.25%, B 2.92%/4.58%, C 4.38%/1.25%, D 3.12%/1.25%. Against held-out PPO, sampling helped the two small-MLP pooled averages but not both larger-MLP arms. All crossed sampled-minus-greedy intervals include zero; neither mode produced final-test timeouts. Primary reporting remains sampled, as frozen. [Paired inference analysis](sampled-minus-greedy.json).

Final validation selected **B / seed 400033 / sampled / step 425984**, with 4/32 heuristic-family wins (12.5%). Its selection was hash-frozen externally before final testing. On the independent 100-board confirmation prefix, including both seats, it scored:

- Base: **8 wins, 192 losses**, 4.0%; board-cluster 95% interval [1.5%, 7.0%].
- Contender: **10 wins, 190 losses**, 5.0%; interval [2.0%, 8.5%].
- Combined: **18/400 wins (4.5%)**, interval [2.25%, 7.0%], no truncations.

This confirms one validation-selected policy, not average B performance. Its lower confirmation rate illustrates the uncertainty of a 32-game validation panel; it is not evidence of regression during training. The first 80 confirmation games already belong to the common test panel; only the remaining 320 are additional games. [Selection](selection.json) · [Confirmation](confirmation-summary.json).

## Strategic and optimization diagnostics

Against base/contender under sampled inference, **66–81% of games ended without a city** and **78–89% without a third settlement build**. Mean final points ranged from 3.54 to 4.15 across arms. These are failures to reach economy milestones before the game ended, not proof of a specific repeated-state loop. The stronger opponent may end the game before later plans become executable.

Knights and maritime trades were actually used. Knight choice/legal-opportunity ratios were A 353/520, B 409/1099, C 167/253 and D 225/448. Trade choice/legal-opportunity ratios were 72–82%; this does **not** show that those trades were productive. Immediate deterministic build/knight wins were selected in 2/7, 3/15, 5/12 and 3/4 opportunities, respectively—too few to rank conversion ability. The diagnostic does not enumerate every possible chance-dependent winning action. [Strategic counts and conditional event timing](strategic-diagnostics.csv).

Training completed 18,647 terminal episodes and 26 administratively truncated episodes over 4,259,840 transitions. Terminal rewards occurred on **0.438% of learner transitions**. Training truncations were A 9, B 9, C 4, D 4; all 26 bounded failure traces were retrieved. Every validation and final evaluation game completed without truncation, illegal action or recorded infrastructure failure. [Training episodes](training-episodes.csv) · [Failure traces](training-failure-traces.json).

The configured base opponent received approximately 24–26% of resets but only **9.6–11.9% of learner transitions**, because its games ended faster. Refreshed policies accumulated roughly 28–30% of resets and 36–40% of transitions against their own snapshots; the neural half initially used the same historical archive as static runs. Six own snapshots were published at absolute 65,536-step intervals. These data support auditing actual learning exposure in the next experiment; they do not justify changing this run's mixture after the fact.

No nonfinite losses occurred. At the final rollout, MLP-128 approximate KL was 0.0027–0.0044 versus 0.0135–0.0204 for MLP-512; clipping fractions were approximately 3.7–4.8% versus 7.6–10.2%. Final critic explained variance was 0.85–0.97, but it is measured against bootstrapped GAE returns, **not independently calibrated eventual-win targets**. High explained variance coexists with weak external play. Sparse rewards produced learning against the historical neural opponent, but little demonstrated progress against the value heuristics. Validation strength versus steps was neither consistently monotonic nor evidence of overtraining; no large checkpoint-selection sweep was performed.

Larger MLPs were not materially slower in this CPU-dominated path: approximately 171–174 learner decisions/s and 353–360 engine actions/s across arms, with about 56 seconds/run spent inside PPO updates. Active-worker telemetry averaged about 4.0% GPU utilization. Sampled heuristic-panel network inference was approximately 1.07–1.18 ms/decision; complete observed decision processing was 1.75–1.92 ms. **These are measurements inside Colab, not browser latency.**

## Scope, recovery and accounting

The [frozen manifest](frozen.json) fixes the schema, rules, masks, PPO settings, seeds, opponent mixture and coverage. MLP-128 has 172,109 parameters; MLP-512 has 884,045. All policies started from scratch, with identical per-seed initial weights across schedules. Terminal rewards remained ±1, gamma 1, rollout 4096, four epochs, minibatch 256, Adam 3e−4, clipping 0.2, GAE lambda 0.95 and entropy coefficient 0.01.

Fresh single/dual pilots measured every condition with its actual inference path. Graph E/F single-worker rates were 54.6/48.7 decisions/s versus roughly 164–178 for the MLP arms. The conservative full matrix fit only 163,840 transitions/run; omitting G fit 180,224. Omitting E/F as well allowed **425,984**, above the requested 393,216 minimum. Two workers did not consistently improve throughput, so one worker was frozen. These omissions occurred before experimental initialization or outcomes. The 200-game teacher corpus and its provenance are retained for a later experiment; its generation cost was 160.9 CPU seconds, and G's pilot optimization cost remains recorded separately. Repaired search was omitted at its preallocation correctness gate.

Coverage completed exactly as frozen:

| Panel | Coverage | Games |
|---|---|---:|
| Validation | Four boards at 0 and first three milestones; eight boards at final step, both final modes | 1,920 |
| Main final test | 20 boards × both seats × three opponents × both modes × ten policies | 2,400 |
| Selected-policy confirmation extension | Additional 80 boards × both seats × base/contender | 320 |
| Weak regression checks | Five boards × both seats × three opponents × ten policies, sampled | 300 |
| Three historical reference policies | Same 20-board main panel, both modes | 720 |

Thus 3,740 distinct final-stage games and 1,920 validation games completed. Weak-opponent and historical rows are context, excluded from the headline heuristic endpoint. Historical sampled family wins were 6/240 and held-out PPO wins 56/120; their different training/preprocessing protocol prevents a causal comparison to these new runs.

The initial attempt failed exact graph CUDA recovery. A CPU cache constructor was found to reseed CUDA and was corrected. The first replacement then exhausted preflight after setup/CLI delays. Those attempts remain preserved in the [earlier failure report](../scaling-20261114/findings.md). Your subsequent recovery instruction authorized one further fault attempt; [the amendment](authorization-amendment.json) records that change. It did not reset the original deadline.

The successful recovery used detached bounded installation, cold/warm-cache continuation checks, graph-first pilots, retained pilot checkpoints and a verified external graph checkpoint upload/resume round trip. All local control tests passed. A later planned handoff restored completed checkpoints and evaluation blocks; no training run was retrained and no learning configuration changed.

Training consumed 415.74 minutes, validation 38.10 minutes and final evaluation 63.01 minutes of manager-accounted phase time. The full original clock, including earlier failures, setup, downtime, transfer and shutdown, was 674.85 minutes. Allocations totaled 658.88 minutes and never overlapped. The requested ten-minute external-copy cadence was a **target, not a perfect achieved guarantee**: during experimental training the median verified-generation gap was 10.76 minutes and the maximum 19.28 minutes because of transfers/retries. All final artifacts were independently reverified. [Resource accounting](resource-summary.json) · [Runtime ledger](runtime-ledger.csv).

## Ranked next scaling direction

1. **Test the prepared opening/economy teacher warm start against its matched MLP-128 PPO control.** Most strong-opponent games end before expansion, and own snapshots did not yield a dependable strength gain. A focused G-versus-B experiment can test whether better early decisions create more useful full-game experience while preserving terminal rewards. Strongest counterevidence: G was omitted here, so its benefit is entirely unestablished; teacher information may not be fully reconstructible from the memoryless policy observation, and copying a heuristic can plateau.
2. **Replicate the static MLP-512 capacity signal with more paired seeds**, if the opening intervention is ineffective or sufficient budget exists. It improved both first-two-seed point estimates at similar elapsed cost, but crossed uncertainty includes zero and absolute heuristic strength remains poor. Do not declare C the winner or deploy it based on this test ranking.
3. **Defer broader graph scaling and a larger GPU.** Graph correctness/recovery now passed, but its collection cost prevented a useful matched-budget test here. Profile CPU opponent/engine work before buying GPU capacity. Graph strength and teacher strength remain unknown; neither has been disproved.

For later deployment experiments, preserve the preselected B/400033 policy as an artifact, not as evidence of expert play. No arm solved negotiation, and PPO probabilities are not calibrated action values or win probabilities.

## Deliverables and reproduction

All nine chart families are provided as PNG and SVG, rendered from saved rows and visually inspected:

1. [Validation versus transitions](01-validation-transitions.png)
2. [Validation versus elapsed time](02-validation-time.png)
3. [Architecture/schedule heatmaps, sampled and greedy](03-factorial-heatmaps.png)
4. [Paired contrasts with seed points and crossed intervals](04-paired-contrasts.png)
5. [Strength versus parameters and Colab decision cost](05-strength-cost.png)
6. [Expansion, action opportunities and timeouts](06-expansion-timeouts.png)
7. [PPO and critic diagnostics](07-ppo-critic.png)
8. [Opponent composition and snapshot ages over training](08-opponent-pools.png)
9. [Utilization, throughput, copies and allocation timeline](09-utilization-recovery.png)

Persistent artifact root: `/root/tinysettler-artifacts/tinysettler-scaling-20261114-a-recovery2/`, outside the synced checkout. Full resumable checkpoints are under `retrieved/runs/ARM-SEED/`; portable inference exports are under `final-policy-exports/`. **All 91 indexed checkpoint/snapshot files passed hash checks**, all ten final checkpoints passed CPU collection/update recovery, and every planned panel passed coverage checks. All 24 executed source hashes, input hashes, 319 objects in the two retained external generations, and 470 latest retrieved files were independently verified. [Verification](verification.json) · [Checkpoint index](checkpoint-index.json) · [Source hashes](source-hashes.json) · [Artifact audit](artifact-audit.json).

Key SHA256 values:

- Executed source archive: `448f737c0bb28289ef6beb362d179854cbf876239bda842a057439d2375a4842`
- Selection manifest: `00526a7e51cea061ca75c78860fef709ade65297a852d0d3999a66e4a45bff52`
- Selected full checkpoint, `runs/B-400033/425984.pt`: `c674a5f17216bdc5c6e1a8a4a636586f1afe56c0637ae98bb058a8be5e8020a7`
- Selected inference export, `final-policy-exports/B-400033.pt`: `e68ab68854a4c3e0740824fb9b3a6f105bf23fe3d07ad34a21cd5b40de235994`

Dependencies remain [locked](../../scaling/requirements-lock.txt), simulator revision `ecf931181b9a65bb4116a2153fb78c16f1438e00`, CLI 0.7.5 and Torch 2.9.1+cu128. The archived trainer/configuration is authoritative; postprocessing only refines labels and derives summaries from saved rows.

Reproduce reports on the persistent CPU host:

```sh
export SCALING_INPUTS=/root/tinysettler-artifacts/tinysettler-scaling-20261114-a-recovery2/inputs
ROOT=/root/tinysettler-artifacts/tinysettler-scaling-20261114-a-recovery2
OUT=/workspace/TinySettler/rl_research/results/scaling-20261114-recovered
PY=/tmp/tinysettler-rl-venv/bin/python
"$PY" "$ROOT/source/sc_report.py" --artifacts "$ROOT" --output "$OUT"
"$PY" rl_research/scaling/postprocess_scaling.py --artifacts "$ROOT" --output "$OUT"
"$PY" "$ROOT/source/sc_verify.py" --artifacts "$ROOT" --output "$OUT"
```

The completed controller resume command preserves the original deadline and exits without allocating:

```sh
"$PY" "$ROOT/source/sc_controller.py" --artifacts "$ROOT" \
  --session tinysettler-scaling-20261114-a --resume
```

The [implementation README](../../scaling/README.md) documents preparation/new-launch and explicit evaluation commands. This experiment is complete; no new allocation is needed. The final delivery manifest identifies the verified source/policy/results package and external full-checkpoint locations. Historical reports, inputs and browser demo remain preserved.
