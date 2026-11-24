# Train/validation split lists

These files list, by dataset file name, exactly which windows were used for
training and which for validation in the paper (random 80/20 split, seed 42).
`train.py` and `journal_experiments/` use them automatically when present.

- `siteA_train.txt`, `siteA_val.txt`: Site A (all models in the paper)
- `siteC_train.txt`, `siteC_val.txt`: Site C (the Site A + Site C experiment only)

File names are relative to the dataset folder (`data/`).
