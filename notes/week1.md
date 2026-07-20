# Week 1 observations

Date: ____  Instance: RunPod Community Cloud, RTX 4090 24GB, $__/hr
Total session cost: $____

## Baseline run
Paste RUN SUMMARY line here:

```
```

nvidia-smi memory during run: ____ GB (vs torch.cuda.memory_allocated: ____ GB)
Why they differ: 

## OOM sweep
| batch | peak GB | s/step | result |
|-------|---------|--------|--------|
| 16    |         |        | ok     |
| 32    |         |        |        |
| 64    |         |        |        |
| 128   |         |        |        |
| 256   |         |        |        |

First OOM at batch = ____  (predicted: ____)

## Bonus: layers sweep (fixed baseline moves, not slope)
| layers | params M | peak GB @ batch 16 |
|--------|----------|--------------------|
| 4      |          |                    |
| 8      |          |                    |

## Prediction vs reality paragraph
(copy final version into predictions.md too)

## Things that surprised me / broke

