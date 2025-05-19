import random

import numpy as np


def seed_worker(worker_id: int):
    """Generate seeds for a PyTorch DataLoader worker.

    This ensures that if multiple workers are sampling randomly, they use
    different sequences of random numbers.

    Args:
        worker_id:
            The ID of the worker process.
    """

    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # print(f"Worker {worker_id} seeded with {worker_seed}")
