# PPO-128 starting-state curriculum experiment

Completed results: [findings](../results/ppo-curriculum-20261111/findings.md).
All six runs and all evaluations finished; runtimes are released. Commands below
document the original launch/recovery workflow, not a request to allocate again.

The variable is the episode starting state. Both arms use exactly the same frozen
50% adapted Catanatron ValueFunction-base, 25% WeightedRandom, 25% legacy heuristic
mixture. `contender` is a held-out configuration of the same heuristic family.
The upstream value functions and weights are unchanged, scored in float64. This
is explicitly an information/RNG adapter, not the unmodified bundled bot.

Three paired starting policies are the first chronological PPO-128 seeds in the
existing manifest: 104729, 130363, 155921, all at 434,176 transitions. Weights and
Adam state are cloned exactly. Pending rollouts are cleared and fresh independent
RNG namespaces start new episodes. All PPO optimization, observations, action
mapping, game rules, terminal rewards and discount remain in the immutable
archived implementation. Only opponents (shared by both arms) and reset schedules
are wired through new subclasses. No rewards for starting points are awarded.

Standard resets are all fresh; curriculum resets are 50% fresh / 25% 4–6 VP /
25% 7–9 VP. All snapshots arise through legal play. Source games are assigned to
training, validation or test before snapshot selection. Full state, historical
counters, development-card age, awards and deck order are retained. Repeated
resets change future dice/steal randomness, not physical facts or remaining
horizon. Policy inputs contain neither snapshot IDs nor hidden state.

The adapter maintains conservative opponent-resource intervals using public
transfers, public hand totals and its own cards. Unknown discarded types are not
read. It resamples opponent resources, hidden development cards/VP and unseen
deck order; it conditions on the observed legal action set and nonterminal fact.
This is an approximate information-state sampler, not an exact Bayesian belief.
Each candidate action is evaluated on the same two worlds and chance streams,
with an independent checkpointed planning RNG. Live RNG is never advanced or
copied into planning. A bounded public-only enumeration fallback is counted;
there is no fallback to the real hidden state. Raw histories are removed from
planning copies. Planning does not use AlphaBeta, SameTurnAlphaBeta or MCTS.

The 14,400-second absolute deadline includes all allocations/retries. Last 1,200
seconds are reserved for retrieval/shutdown. At most two replacements are allowed (one following the user-requested pause,
one for a technical failure), only after confirmed termination and preserving
the original deadline. Connection/transfer retries are bounded at six each
across the experiment, with backoff; verified snapshot copies run every two
minutes. This controller-only recovery amendment is recorded in
`technical-recovery.json`; the executed training bundle is unchanged. Atomic checkpoints
are written at least every five minutes and common intermediate/final steps;
the controller hash-verifies external copies. Completed/incomplete runs are not
silently restarted. The controller's independent watchdog stops the dedicated
runtime before the hard deadline. Archived historical files are never modified.

## Commands

Run from repository root. The original dependency lock is
[requirements-lock.txt](../overnight/requirements-lock.txt).

```sh
export PYTHONHASHSEED=0
export PYTHONPATH=rl_research/overnight:rl_research/reliability:rl_research/curriculum
/tmp/tinysettler-rl-venv/bin/python rl_research/curriculum/prepare_data.py \
  --out /tmp/tinysettler-curriculum-data
/tmp/tinysettler-rl-venv/bin/python -m pytest -q \
  rl_research/overnight/test_contract.py rl_research/curriculum/test_curriculum.py
/tmp/tinysettler-rl-venv/bin/python rl_research/curriculum/prepare_bundle.py \
  --data /tmp/tinysettler-curriculum-data \
  --artifacts /root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a
python rl_research/curriculum/curriculum_controller.py \
  --session tinysettler-ppo-curriculum-20261111-a \
  --artifacts /root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a
```

Resume the same controller with `--resume`. A terminated runtime can be replaced
within that bounded allowance with `--resume --replace`. Never change the artifact root to extend a budget.
For unattended launch, use `subprocess.Popen(..., start_new_session=True)` with
logs in the artifact root. Existing authentication and quota are used.

Evaluation on CUDA from the extracted executed source (the original loader
expects CUDA checkpoint tensors). Set the deadline to the authorized evaluation
deadline and use a new output path to execute games rather than read cached rows:

```sh
PYTHONHASHSEED=0 python curriculum_evaluate.py CHECKPOINT.pt \
  --seed-start 880000000000 --boards 20 \
  --deadline ABSOLUTE_UNIX_DEADLINE --output evaluation.json
```

Those test seeds are now consumed; replay reproduces the completed test protocol,
not a new untouched test. The source directory is
`/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a/source/` and
final checkpoints are `retrieved/ARM-SEED/176128.pt` under that artifact root.
Use that archived source for provenance-checked loading; repository-only reporting
tools were added after the training bundle was frozen.

Rebuild reports and verify external checkpoints on CPU without a runtime:

```sh
PYTHONHASHSEED=0 /tmp/tinysettler-rl-venv/bin/python - <<'PY'
import runpy, sys
from pathlib import Path
repo = Path('/workspace/TinySettler')
root = Path('/root/tinysettler-artifacts/tinysettler-ppo-curriculum-20261111-a')
out = repo / 'rl_research/results/ppo-curriculum-20261111'
sys.path.insert(0, str(root / 'source'))
for script in ['curriculum_report.py', 'verify_results.py']:
    sys.argv = [script, '--artifacts', str(root), '--output', str(out)]
    runpy.run_path(str(repo / 'rl_research/curriculum' / script), run_name='__main__')
sys.argv = ['finalize_report.py', '--output', str(out)]
runpy.run_path(str(repo / 'rl_research/curriculum/finalize_report.py'), run_name='__main__')
PY
```

Coverage and the common additional-transition target are frozen after fresh
pilots of both arms and strong-opponent evaluations, including an untrained
policy. No cached results are used for timing. Primary evaluation is greedy,
with both seats against each of five opponents; diagnostic sampled continuations
use independent policy RNG. Evaluation preserves all training RNG streams.

Historical failure records did not archive the original Python hash seed or
full game trace. Exact reconstruction is reported only when the outcome and
all action counts match. Same-board approximate reconstructions are explicitly
labeled and are diagnostic-only. Repeated hashes do not establish loops; legal
opportunities and action probabilities are recorded. PPO probabilities are not
Q-values or calibrated win probabilities.

CLI syntax follows installed help and the
[official Colab CLI documentation](https://github.com/googlecolab/google-colab-cli).
[Colab runtime limits](https://research.google.com/colaboratory/faq.html) still apply.
