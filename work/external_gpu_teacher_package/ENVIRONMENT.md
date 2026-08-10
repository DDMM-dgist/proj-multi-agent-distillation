# Environment requirements — EXTERNAL GPU teacher fine-tune (PC-A1)

The warm-start checkpoint (`base_teacher/best.ckpt`) was produced with the stack
below. **Reproduce it exactly** — a 0.16.x nequip will not load a 0.15 checkpoint
cleanly, and a mismatched allegro/e3nn will silently mis-instantiate the model.

| package    | version                        | notes |
|------------|--------------------------------|-------|
| python     | 3.10–3.11                      | |
| nequip     | **0.15.0**                     | base training stack; NOT 0.16.x |
| allegro    | **0.7.1**                      | |
| torch      | **2.6.0** (+cu124)             | CUDA 12.4 build |
| lightning  | matching the 0.15 era (pytorch-lightning ≥2.x) | |
| e3nn       | matching allegro 0.7.1         | |
| ase        | ≥3.22                          | reads the extxyz corpus |
| wandb      | any (run offline: `WANDB_MODE=offline`) | |

Suggested:
```bash
conda create -n allegro015 python=3.11 -y && conda activate allegro015
pip install nequip==0.15.0 allegro==0.7.1
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# then verify:
python -c "import nequip,torch;print(nequip.__version__,torch.__version__,torch.cuda.is_available())"
```

## Known environment gotcha (carried from the v6 handoff)
This allegro build **silently returns garbage (no error)** if the model runs on a
non-zero CUDA index while other GPUs remain visible. Always mask to one device:
`export CUDA_VISIBLE_DEVICES=<physical_idx>` and use `cuda:0`. SLURM cgroup masking
already handles this for batch jobs. `scripts/run_training.sh` sets
`CUDA_VISIBLE_DEVICES=0` by default.
