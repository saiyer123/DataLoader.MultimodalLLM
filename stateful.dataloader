import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from typing import Iterator, Optional, Dict, Any, List

class StatefulSampler(Sampler[int]):    
    def __init__(
        self, 
        data_source: Dataset, 
        shuffle: bool = True, 
        seed: int = 0
    ):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Current position in the indices list (how many samples we've yielded)
        self._current_position = 0
        
        # The order of indices for this epoch
        self._indices: Optional[List[int]] = None
        
        # Generate initial indices
        self._generate_indices()
    
    def _generate_indices(self) -> None:
        """Generate the index order for the current epoch."""
        n = len(self.data_source)
        
        if self.shuffle:
            # Create a generator with deterministic seed based on epoch
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            
            # Generate shuffled indices
            self._indices = torch.randperm(n, generator=generator).tolist()
        else:
            # Sequential order
            self._indices = list(range(n))
    
    def __iter__(self) -> Iterator[int]:
        """
        Iterate over indices, starting from current position.
        """
        # Yield remaining indices from current position
        while self._current_position < len(self._indices):
            idx = self._indices[self._current_position]
            self._current_position += 1
            yield idx
        
        # Reset for next epoch (if training continues)
    
    def __len__(self) -> int:
        """Return number of remaining samples in the current epoch."""
        return len(self._indices) - self._current_position
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number for shuffling.
        This should be called at the start of each epoch to ensure
        different shuffle orders across epochs.
        """
        self.epoch = epoch
        self._current_position = 0
        self._generate_indices()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the current state of the sampler.
        """
        return {
            'indices': self._indices.copy() if self._indices else None,
            'current_position': self._current_position,
            'epoch': self.epoch,
            'seed': self.seed,
            'shuffle': self.shuffle,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore sampler state from a checkpoint.
        
        Args:
            state_dict: State dictionary from a previous state_dict() call
        """
        self._indices = state_dict['indices'].copy() if state_dict['indices'] else None
        self._current_position = state_dict['current_position']
        self.epoch = state_dict['epoch']
        self.seed = state_dict['seed']
        self.shuffle = state_dict['shuffle']
    
    def remaining_samples(self) -> int:
        """Return the number of samples remaining in the current epoch."""
        if self._indices is None:
            return 0
        return len(self._indices) - self._current_position
    
    def reset(self) -> None:
        """Reset position to start of current epoch (keeps same shuffle order)."""
        self._current_position = 0


class StatefulDataLoader:
    """
    This class wraps PyTorch's DataLoader and adds state management capabilities,
    enabling training to resume from the exact sample where it was interrupted.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        collate_fn=None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Create our stateful sampler
        self.sampler = StatefulSampler(
            data_source=dataset,
            shuffle=shuffle,
            seed=seed
        )
        
        # Track samples processed in current iteration
        self._samples_yielded_in_epoch = 0
        
        # Create the PyTorch DataLoader
        self._dataloader = self._create_dataloader()
    
    def _create_dataloader(self) -> DataLoader:
        """Create the PyTorch DataLoader with our sampler."""
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
    
    def __iter__(self):
        """
        Iterate over batches.
        Yields batches starting from the current position in the sampler.
        """
        self._samples_yielded_in_epoch = 0
        if self.num_workers != 0:
            raise ValueError(
                "StatefulDataLoader currently supports num_workers=0 only."
            )

        for batch in self._dataloader:
            # Track how many samples we've yielded
            # For the last batch, it might be smaller if drop_last=False
            if isinstance(batch, dict) and 'input_ids' in batch:
                batch_size = batch['input_ids'].shape[0]
            elif isinstance(batch, (list, tuple)):
                batch_size = len(batch[0]) if batch else 0
            else:
                batch_size = self.batch_size
            
            yield batch
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._dataloader)
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number for shuffling.
        
        Should be called at the start of each epoch.
        
        Args:
            epoch: The current epoch number
        """
        self.sampler.set_epoch(epoch)
        self._samples_yielded_in_epoch = 0
        # Recreate dataloader to pick up new sampler state
        self._dataloader = self._create_dataloader()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the complete state needed for resumption.
        
        Returns:
            Dictionary containing:
                - sampler_state: State of the StatefulSampler
                - batch_size: Batch size (for validation)
                - samples_yielded: Samples processed so far
        """
        return {
            'sampler_state': self.sampler.state_dict(),
            'batch_size': self.batch_size,
            'samples_yielded_in_epoch': self._samples_yielded_in_epoch,
            'drop_last': self.drop_last,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore DataLoader state from a checkpoint.
        
        Args:
            state_dict: State dictionary from a previous state_dict() call
        
        Raises:
            ValueError: If batch_size doesn't match (could cause issues)
        """
        # Validate batch size matches
        if state_dict['batch_size'] != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: checkpoint has {state_dict['batch_size']}, "
                f"but DataLoader has {self.batch_size}. "
                "Resuming with different batch size is not supported."
            )
        
        # Restore sampler state
        self.sampler.load_state_dict(state_dict['sampler_state'])
        self._samples_yielded_in_epoch = state_dict.get('samples_yielded_in_epoch', 0)
        
        # Recreate dataloader to pick up restored sampler state
        self._dataloader = self._create_dataloader()
    
    def remaining_samples(self) -> int:
        """Return number of samples remaining in current epoch."""
        return self.sampler.remaining_samples()
    
    @property
    def current_position(self) -> int:
        """Return current sample position in the epoch."""
        return self.sampler._current_position


# Convenience function for easy integration
def create_stateful_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
    collate_fn=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> StatefulDataLoader:
    """
    Create a StatefulDataLoader with the given parameters.
    """
    return StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # Simple test to verify the implementation works
    from torch.utils.data import TensorDataset
    
    print("Testing StatefulDataLoader...")
    
    # Create a simple dataset
    data = torch.arange(100)
    dataset = TensorDataset(data)
    
    # Create stateful dataloader
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True,
        seed=42
    )
    
    # Iterate through some batches
    print("\n--- First run (partial) ---")
    batches_seen = []
    for i, batch in enumerate(dataloader):
        batches_seen.append(batch[0].tolist())
        print(f"Batch {i}: {batch[0][:3].tolist()}...")  # Print first 3 elements
        
        if i == 4:  # Stop after 5 batches
            # Save state
            state = dataloader.state_dict()
            print(f"\nSaving state at batch {i}, position {dataloader.current_position}")
            break
    
    # Create new dataloader and restore state
    print("\n--- Resuming from checkpoint ---")
    dataloader2 = StatefulDataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True,
        seed=42
    )
    dataloader2.load_state_dict(state)
    
    print(f"Restored position: {dataloader2.current_position}")
    print(f"Remaining samples: {dataloader2.remaining_samples()}")
    
    # Continue iteration
    resumed_batches = []
    for i, batch in enumerate(dataloader2):
        resumed_batches.append(batch[0].tolist())
        print(f"Resumed batch {i}: {batch[0][:3].tolist()}...")
    
    print("\n--- Verification ---")
    print(f"Batches before save: {len(batches_seen)}")
    print(f"Batches after resume: {len(resumed_batches)}")
    print(f"Total batches: {len(batches_seen) + len(resumed_batches)}")
    print(f"Expected batches: {len(dataloader)}")
    
    # Verify no overlap
    all_samples = []
    for b in batches_seen + resumed_batches:
        all_samples.extend(b)
    
    unique_samples = set(all_samples)
    print(f"Unique samples seen: {len(unique_samples)}")
    print(f"Total samples: {len(dataset)}")
    
    if len(unique_samples) == len(dataset) and len(all_samples) == len(dataset):
        print("\n✓ SUCCESS: All samples seen exactly once!")
    else:
        print("\n✗ FAILURE: Sample coverage issue detected")
