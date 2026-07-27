# Week 1 predictions — fill in BEFORE renting the GPU

Rules: commit this file before the first GPU run. Wrong is fine; unstated is not.

## Model config (defaults)
d_model=512, nhead=8, num_layers=4, ffn=2048, vocab=10000, seq_len=128

## 1. Parameter count
Work it out roughly (embed + pos + 4x encoder layers + head):

- Predicted params: 128 M
- Predicted param memory (fp32, params x 4 bytes): 512 MB

## 2. Training state (no activations)
params + grads + AdamW momentum + variance, all fp32 = 4x param memory:

- Predicted training state: 2 GB

## 3. Peak memory at batch=16
Training state + activations + allocator/CUDA overhead. Take a guess:

- Predicted peak: 8 GB

## 4. OOM point on a 24 GB RTX 4090
Batch size doubles: 16, 32, 64, 128, 256...

- Predicted first batch size that OOMs: 64

## 5. One sentence: why does peak memory scale with batch size but
##    "training state" doesn't?

training parameter gradients moments do not scale with batch size.

---

## After the run — the paragraph that matters

Predicted peak was ____, measured peak was ____. The gap comes from:

(write it in your own words — this is the interview answer)
