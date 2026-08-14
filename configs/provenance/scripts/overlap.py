import json

with open("/tmp/kisti_verify/cat_indices.json") as f:
    mapping = json.load(f)
cat_indices = mapping["cat_indices"]

with open("/tmp/kisti_verify/split_indices.json") as f:
    split_indices = json.load(f)
train_set = set(split_indices["train"])
val_set = set(split_indices["val"])
test_set = set(split_indices["test"])
assert len(train_set & val_set) == 0
assert len(train_set & test_set) == 0
assert len(val_set & test_set) == 0
assert len(train_set) + len(val_set) + len(test_set) == 11424

manifest_path = "/home/hyunjin/distill-real-user/configs/runs/sio2-sox-allegro-simplenn-r11/teacher_baseline_slice_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

records = manifest["records"]
print("n records:", len(records))

counts = {"train": 0, "val": 0, "test": 0, "unresolved": 0}
unresolved_examples = []
by_slice = {}
for rec in records:
    cat = rec["source_category"]
    local_idx = rec["source_local_index"]
    memberships = rec.get("memberships", [])
    gidx = None
    if cat in cat_indices and 0 <= local_idx < len(cat_indices[cat]):
        gidx = cat_indices[cat][local_idx]
    if gidx is None:
        counts["unresolved"] += 1
        unresolved_examples.append(rec)
        continue
    if gidx in train_set:
        which = "train"
    elif gidx in val_set:
        which = "val"
    elif gidx in test_set:
        which = "test"
    else:
        which = None
    counts[which] += 1
    for m in memberships:
        by_slice.setdefault(m, {"train": 0, "val": 0, "test": 0})
        by_slice[m][which] += 1

print("Overlap counts:", counts)
print("By deployment slice:", json.dumps(by_slice, indent=2))
print("unresolved examples:", unresolved_examples[:3])
