import torch
from typing import Optional

from transformers import Trainer, TrainingArguments

from torch.utils.data import Sampler
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed

BATCH_PER_DEVICE=2
GRAD_ACC=3
# 2*4*3 // 8 = 3

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source ,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count





# Dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([idx]),
            "labels": torch.tensor([idx]),
        }

# Dummy model
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, input_ids=None, labels=None):
        outputs = self.linear(input_ids.float())
        return {"logits": outputs}


class MyTrainer(Trainer):
    def _get_train_sampler(self):
        effective_batch_size = BATCH_PER_DEVICE * self.accelerator.num_processes * GRAD_ACC
        print("effective_batch_size:", effective_batch_size)
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=8,
            batch_size=effective_batch_size//8,
            repeat_count=1*GRAD_ACC,
            seed=self.args.seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        all_labels = self.accelerator.gather(labels)

        rank = self.accelerator.process_index 
        if rank == 0:
            print("rank:", self.accelerator.process_index, "global_step:", self.state.global_step, "labels:", all_labels.tolist(), "\n")

        # Forward pass
        outputs = model(input_ids)

        logits = outputs['logits']
        loss = torch.nn.functional.mse_loss(logits, labels.float())

        if return_outputs:
            return loss, outputs
        return loss    

# Run dummy training
if __name__ == "__main__":
    dataset = DummyDataset(size=300)
    model = DummyModel()

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_PER_DEVICE*GRAD_ACC,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
        logging_dir="./logs",
        disable_tqdm=False,
        seed=42,
        report_to=None
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
