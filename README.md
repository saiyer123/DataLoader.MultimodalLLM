# StatefulDataLoader — Mid-Epoch Training Resumption for PyTorch

A drop-in PyTorch DataLoader replacement that saves iteration state so training can resume from the **exact sample** where it was interrupted — no duplicated samples, no skipped samples, no broken shuffle order.

---

## The Problem

PyTorch's default `DataLoader` has no checkpointing. When training is interrupted (cluster preemption, OOM crash, manual kill), the next run re-shuffles and restarts from sample 0. This causes:

- **Duplicated samples** — early samples re-processed before the model sees later ones
- **Skipped samples** — late-epoch samples never seen if interruptions are frequent
- **Non-reproducible training** — different effective data order every run

Model weights and optimizer state are routinely checkpointed. Data loading state is not — this fixes that.

---

## Solution

Two pieces of information are sufficient to resume data loading exactly:

1. The **shuffled index order** for the current epoch (generated deterministically from a seed)
2. The **current position** in that order (how many samples have been yielded)

`StatefulSampler` generates and owns the index order, exposes `state_dict()` / `load_state_dict()`, and resumes iteration from any position. `StatefulDataLoader` wraps PyTorch's `DataLoader` with this sampler and propagates checkpoint save/load to it.

---

## Usage

**Basic setup**
```python
from stateful_dataloader import StatefulDataLoader

loader = StatefulDataLoader(
    dataset=my_dataset,
    batch_size=32,
    shuffle=True,
    seed=42,
)
```

**Save state mid-epoch**
```python
for step, batch in enumerate(loader):
    train_step(batch)

    if step % save_every == 0:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dataloader": loader.state_dict(),   # <-- save iteration state
        }
        torch.save(checkpoint, "checkpoint.pt")
```

**Resume from checkpoint**
```python
loader = StatefulDataLoader(dataset=my_dataset, batch_size=32, shuffle=True, seed=42)

checkpoint = torch.load("checkpoint.pt")
loader.load_state_dict(checkpoint["dataloader"])  # restores exact position

# continues from the next unprocessed sample
for batch in loader:
    train_step(batch)
```

**Per-epoch shuffle (multi-epoch training)**
```python
for epoch in range(num_epochs):
    loader.set_epoch(epoch)   # new shuffle order, position reset to 0
    for batch in loader:
        train_step(batch)
```

---

## API

### `StatefulDataLoader`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `Dataset` | — | PyTorch Dataset |
| `batch_size` | `int` | `1` | Samples per batch |
| `shuffle` | `bool` | `True` | Shuffle at start of each epoch |
| `seed` | `int` | `0` | Base seed for deterministic shuffling |
| `collate_fn` | `callable` | `None` | Custom collation (required for multimodal/variable-length data) |
| `num_workers` | `int` | `0` | Must be `0` (see Limitations) |
| `drop_last` | `bool` | `False` | Drop incomplete final batch |

| Method | Description |
|---|---|
| `state_dict()` | Returns serializable dict with sampler state + position |
| `load_state_dict(state)` | Restores exact position; raises if batch_size mismatches |
| `set_epoch(epoch)` | Advances to new epoch with fresh shuffle; resets position |
| `remaining_samples()` | Samples left in current epoch |
| `current_position` | Samples yielded so far this epoch |

---

## How It Works

At the start of each epoch, `StatefulSampler` generates a complete shuffled index list using a deterministic seed (`base_seed + epoch`). As iteration proceeds, a single integer counter tracks how many samples have been yielded. This counter implicitly encodes both batch number and position within the batch — batches are just contiguous slices of the index list.

On checkpoint, the index list and counter are serialized. On resume, the same list is restored and the sampler begins yielding from `indices[counter:]`. No reshuffling occurs, so the data order is identical to an uninterrupted run.

State is tracked at the **sample level** rather than batch level, which handles partial batches and `drop_last` naturally without additional bookkeeping.

---

## Tests

`test_resume.py` verifies two properties:

1. **No duplication or omission** — combined sample sequence before + after interruption contains each index exactly once
2. **Deterministic equivalence** — interrupted-and-resumed sequence is bit-for-bit identical to an uninterrupted run under the same seed

```bash
python test_resume.py
# PASS: Resume correctness test
```

---

## Limitations

- **Single-worker only** (`num_workers=0`) — multi-worker DataLoader uses subprocess-based prefetching whose internal state is not exposed; extending statefulness to that setting requires per-worker coordination outside PyTorch's public API.
- **Variable-length / multimodal batches** require a custom `collate_fn` — PyTorch's default collation cannot batch sequences or images of different sizes. This is orthogonal to statefulness and is handled downstream by tokenizers and image processors in real pipelines.

---

## Requirements

```
torch>=2.0
torchvision>=0.15
datasets>=2.14
numpy<2
```

```bash
pip install -r requirements.txt
```
