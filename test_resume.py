from stateful_dataloader import StatefulDataLoader
from torch.utils.data import TensorDataset
import torch

def test_resume():
    data = TensorDataset(torch.arange(30))
    loader1 = StatefulDataLoader(data, batch_size=1, shuffle=True, seed=42)

    seen_before = []
    it = iter(loader1)
    for _ in range(10):
        seen_before.append(next(it)[0].item())

    state = loader1.state_dict()

    loader2 = StatefulDataLoader(data, batch_size=1, shuffle=True, seed=42)
    loader2.load_state_dict(state)

    seen_after = seen_before + [x[0].item() for x in loader2]

    loader3 = StatefulDataLoader(data, batch_size=1, shuffle=True, seed=42)
    uninterrupted = [x[0].item() for x in loader3]

    assert seen_after == uninterrupted
    print("PASS: Resume correctness test")

if __name__ == "__main__":
    test_resume()
