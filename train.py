"""Week 1: minimal transformer training loop with memory instrumentation.

Usage (week 1):
    python train.py                          # baseline: batch 16, 4 layers
    python train.py --batch-size 32          # OOM sweep: 32, 64, 128, ...
    python train.py --num-layers 8           # watch the fixed baseline move

Week 2 levers (leave off for week 1):
    python train.py --amp                    # mixed precision (autocast + GradScaler)
    python train.py --ckpt                   # activation checkpointing

Every run prints a one-line summary at the end — paste it into notes/week1.md.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

VOCAB = 10_000
SEQ_LEN = 128


class SmallModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, use_ckpt=False):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(512, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, VOCAB)
        self.use_ckpt = use_ckpt

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos(positions)
        if self.use_ckpt:
            # Week 2: recompute activations in backward instead of storing them.
            # ~30% extra compute for a large activation-memory saving.
            for mod in self.transformer.layers:
                x = torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=False)
        else:
            x = self.transformer(x)
        return self.head(x)


def gb(n_bytes: int) -> float:
    return n_bytes / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--n-seqs", type=int, default=1000, help="fake dataset size")
    p.add_argument("--amp", action="store_true", help="week 2: mixed precision")
    p.add_argument("--ckpt", action="store_true", help="week 2: activation checkpointing")
    p.add_argument("--smoke", action="store_true", help="tiny CPU-friendly config, 2 batches")
    args = p.parse_args()

    if args.smoke:
        args.d_model, args.num_layers, args.batch_size, args.n_seqs = 64, 2, 4, 16

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else " (smoke test only — rent the GPU)"))

    model = SmallModel(args.d_model, num_layers=args.num_layers, use_ckpt=args.ckpt).to(device)

    n_params = sum(par.numel() for par in model.parameters())
    param_bytes = sum(par.numel() * par.element_size() for par in model.parameters())
    print(f"params: {n_params/1e6:.1f}M | param memory: {param_bytes/1e6:.1f} MB (fp32)")
    print(f"predicted training state (params+grads+AdamW m+v = 4x): {gb(4*param_bytes):.3f} GB")
    print("  (activations NOT included — that's the gap you're here to measure)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and cuda)
    loss_fn = nn.CrossEntropyLoss()

    data = torch.randint(0, VOCAB, (args.n_seqs, SEQ_LEN))
    loader = DataLoader(
        TensorDataset(data, data), batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=cuda,
    )

    if cuda:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    steps = 0
    for epoch in range(args.epochs):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.amp and cuda):
                outputs = model(inputs)
                loss = loss_fn(outputs.view(-1, VOCAB), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            steps += 1

            if batch_idx % 20 == 0:
                mem = f" | alloc {gb(torch.cuda.memory_allocated()):.2f} GB, peak {gb(torch.cuda.max_memory_allocated()):.2f} GB" if cuda else ""
                print(f"epoch {epoch} batch {batch_idx:4d} loss {loss.item():.4f}{mem}")
            if args.smoke and batch_idx >= 1:
                break

    if cuda:
        torch.cuda.synchronize()  # kernels are async — without this, timing lies
    dt = time.time() - t0

    print("\n=== RUN SUMMARY (paste into notes/week1.md) ===")
    summary = (
        f"batch={args.batch_size} layers={args.num_layers} d_model={args.d_model} "
        f"amp={args.amp} ckpt={args.ckpt} | params={n_params/1e6:.1f}M | "
        f"steps={steps} | {dt/max(steps,1):.3f}s/step"
    )
    if cuda:
        summary += (
            f" | alloc={gb(torch.cuda.memory_allocated()):.2f}GB"
            f" peak={gb(torch.cuda.max_memory_allocated()):.2f}GB"
        )
    print(summary)


if __name__ == "__main__":
    main()
