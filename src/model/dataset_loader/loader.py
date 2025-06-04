import torch
from config import CFG, char2id
from speech_dataset import SpeechDataset, _collate_fn
from torch.utils.data import DataLoader, random_split

train_dataset = SpeechDataset(char2id, split=CFG.dataset_list[4], max_len=0)
generated_dataset = SpeechDataset(char2id, split=CFG.dataset_list[0], augmentation=True, max_len=0)
test_dataset = SpeechDataset(char2id, split=CFG.dataset_list[2], max_len=0)
augmented_dataset = torch.utils.data.ConcatDataset([train_dataset, generated_dataset])

train_size: int = int(0.8 * len(augmented_dataset))
val_size: int = len(augmented_dataset) - train_size

train_data, val_data = random_split(
    augmented_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(1),
)

print(
    "Data prepared",
    f"Train      data length: {len(train_data)}",
    f"Validation data length: {len(val_data)}",
    f"Test       data length: {len(test_dataset)}",
    sep="\n",
)

train_dataloader = DataLoader(
    train_data,
    batch_size=CFG.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    collate_fn=_collate_fn,
    num_workers=CFG.worker,
)

val_dataloader = DataLoader(
    val_data,
    batch_size=CFG.batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    collate_fn=_collate_fn,
    num_workers=CFG.worker,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    collate_fn=_collate_fn,
    num_workers=CFG.worker,
)

print(
    "DataLoaders prepared",
    f"Train      data length: {len(train_dataloader)}",
    f"Validation data length: {len(val_dataloader)}",
    f"Test       data length: {len(test_dataloader)}",
    sep="\n",
)
