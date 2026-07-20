# ml-infra-lab

Artifact trail for a 6-month ML infrastructure prep plan. Every experiment
here ran on rented GPUs; predictions were committed before measurements.

- Phase 1 (months 1-2): training loop memory anatomy, DDP, FSDP
- Phase 2 (months 3-4): vLLM serving with measured throughput
- Phase 3 (months 5-6): design drills + packaging

## Week 1 — memory anatomy of training
- `train.py` — instrumented transformer training loop
- `predictions.md` — numbers committed before first GPU run
- `notes/week1.md` — measured results, OOM sweep
- `cheatsheet.md` — pod session commands

Workflow rule: predict, measure, explain the gap, terminate the pod.
