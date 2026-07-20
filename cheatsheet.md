# Pod session cheatsheet — don't google on the clock

## Launch
Template: RunPod PyTorch 2.x | GPU: RTX 4090 | Community Cloud, on-demand

## First 60 seconds on the pod
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    git clone https://github.com/<you>/ml-infra-lab && cd ml-infra-lab

## Run (terminal 1)
    python train.py                     # baseline
    python train.py --batch-size 32    # then 64, 128, 256 until OOM
    python train.py --num-layers 8     # bonus

## Watch (terminal 2)
    watch -n 1 nvidia-smi
    nvidia-smi dmon -s um              # per-second utilization + memory

## Teardown — NON-NEGOTIABLE
1. git add -A && git commit -m "week1 numbers" && git push
2. Terminate the pod (not just stop — stopped pods bill storage)
3. Check billing page shows $0/hr running
4. Spend alert set at $20
