import json, torch

with open("/tmp/kisti_verify/cat_indices.json") as f:
    mapping = json.load(f)
cat_indices = mapping["cat_indices"]
total = mapping["total_frames"]
assert total == 11424

# Exact reproduction of nequip.data.dataset.utils.RandomSplitAndIndexDataset (nequip 0.15.0/0.16.1, read verbatim):
#   generator = torch.Generator().manual_seed(seed)
#   subset_names = list(split_dict.keys())   # order as declared in config.yaml: train, val, test
#   lengths = [split_dict[name] for name in subset_names]
#   splits = torch.utils.data.random_split(dataset, lengths, generator=generator)
seed = 123
split_dict = {"train": 0.8, "val": 0.1, "test": 0.1}   # exact order from .hydra/config.yaml
subset_names = list(split_dict.keys())
lengths = [split_dict[name] for name in subset_names]

generator = torch.Generator().manual_seed(seed)
dummy = list(range(total))  # only __len__/indexing needed to compute the split; real dataset content irrelevant to index assignment
splits = torch.utils.data.random_split(dummy, lengths, generator=generator)

split_indices = {}
for name, subset in zip(subset_names, splits):
    split_indices[name] = set(subset.indices)
    print(f"{name}: n={len(subset.indices)}")

with open("/tmp/kisti_verify/split_indices.json", "w") as f:
    json.dump({k: sorted(v) for k, v in split_indices.items()}, f)

print("torch version:", torch.__version__)
