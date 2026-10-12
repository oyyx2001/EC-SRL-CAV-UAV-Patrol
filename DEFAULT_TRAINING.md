# Optional CS-RDQN training

**Use pretrained evaluation in README.md first.** Training is optional and was not run during release preparation.

Public entries are `configs/case1_xinjiekou.yaml` and `configs/case2_baijiahu.yaml`; old filenames remain compatibility aliases. Both set 300,000 decision transitions, seed 42, candidate-aware Rainbow DQN, 25 action slots and 38 candidate features. The unchanged main loop retains Double DQN, dueling heads, masked targets, NoisyNet after 1,000 warm-up transitions, PER and elapsed-time SMDP discounting (gamma 0.99, reference interval 60 s).

Both training and rollout evaluation require authorized matching scene inputs. Geographic data/derivatives are excluded from the public index pending permission checks; local files and config paths are preserved. Run `python -m tools.check_data_assets` before the following optional commands. Code and original weights are MIT-licensed, not the geographic inputs.

```powershell
python -m rl.train_masked_dqn --config configs/case1_xinjiekou.yaml --algo rainbow_dqn --seed 42 --timesteps 300000 --disable-tensorboard
python -m rl.train_masked_dqn --config configs/case2_baijiahu.yaml --algo rainbow_dqn --seed 42 --timesteps 300000 --disable-tensorboard
```

New outputs go under `results/`, not `checkpoints/`. Release models are original final 300k policies, not newly trained or validation-best substitutes. They omit optimizer state and are evaluation artifacts, not resumable training snapshots. Resume training only from a corresponding unsanitized training checkpoint.

Historical `configs/ablations/formal/*_full.yaml` files are unavailable. Original metadata and run summaries establish the training seed/budget, architecture, replay settings and reward signature. Public YAMLs preserve the audited current runtime and reproduce the original models' Table VIII evaluation. This does not certify that retraining reproduces original weights bitwise.

Shared compatibility hooks remain where deleting them could affect the main simulator. The public factory/trainer exposes only `rainbow_dqn` / `candidate_aware`, not complete standalone baseline/ablation variants.
