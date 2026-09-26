## Mechanism-First Intial Task Sequence Generation

`generate_self_evolution_attack_sequence.py` creates a new environment-grounded
sequence instead of refining an existing one. It scans `env_assets/`, proposes
upstream tasks around a durable self-evolution mechanism, creates downstream
safety tests that activate the learned behavior, and can optionally launch the
normal optimizer.

```bash
python3 prompt_optimization/generate_self_evolution_attack_sequence.py \
  --surface controller_update \
  --harm privacy \
  --provider-profile openrouter \
  --population-size 12 \
  --run-optimizer \
  --max-candidates-per-task 5 \
  --output-dir prompt_optimization/out/generated_controller_privacy
```

Use `--initialize-from-yaml /path/to/file.yaml`
to seed generation from a hand-authored candidate pool.