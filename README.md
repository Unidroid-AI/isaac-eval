<p align="center">
  <strong>UNIDROID</strong><br>
  Training general and custom robot brains through challenges.
</p>

<h1 align="center">isaac-eval</h1>

<p align="center">
  The Isaac Lab harness that runs a challenge submission on a Nepher EnvHub benchmark and scores it.
</p>

Covers the `task-*` campaigns: navigation, manipulation, and more. Validators read the score from `evaluation_result.json`.

## Requirements

```
Isaac Sim 6.1
Isaac Lab 3.0
nepher          # downloads the Nepher EnvHub assets
task-*          # the campaign package being evaluated
brain-comm      # only for runtime: brain
```

## Install

```bash
pip install nepher
nepher download <env_ids...>

cd isaac-eval && pip install -e .
cd ../<task-*> && pip install -e .
```

## Run

```bash
isaac-eval --config configs/task-franka-tabletop.yaml --viz none
```

`python scripts/evaluate.py` is the same command. `--viz none` runs without a window. Isaac Lab 3 has no `--headless` flag.

The config picks the runtime. `runtime: brain` starts one Isaac process per GPU and connects each to a brain socket under `brain.socket_dir`. `placement: paired` puts worker `i` and `brain-i.sock` on GPU `i`. `placement: split` gives the first half of the GPUs to brains. Any other runtime loads a checkpoint into the Isaac process.

Campaign configs live in [configs/](configs/).

### Pre-fetched bundles

By default the benchmark environment comes from the Nepher cache, which downloads it from EnvHub. A validator that fetched the benchmark from Unidroid BenchHub passes the extracted bundle instead:

```bash
isaac-eval --config configs/task-franka-tabletop-v13.yaml --viz none --bundle-dir /benchmark/franka-tabletop
```

`ISAAC_EVAL_BUNDLE_DIR` works the same way. The bundle must hold the EnvHub `manifest.yaml`, and its `id` must match `benchmark_env_id` in the config. Nothing is downloaded and no Nepher API key is needed. A read-only mount is fine: if the folder is not named after the env id, a symlink under a temporary cache directory points to it.

## Results

```
<log_dir>/
  results.json              # per-episode records
  summary.txt               # printed at the end of the run
  config.yaml               # the resolved config
  evaluation_result.json    # the score validators read
```

A checkpoint run writes the first three files to `<log_dir>/eval_run_<timestamp>/` and `evaluation_result.json` to the working directory. `--result-path` sets another location for `evaluation_result.json` in either runtime.

## License

BSD-3-Clause. Copyright (c) 2026 Unidroid. See [LICENSE](LICENSE).
