import torch

def default_collate(samples):
    batch = dict()
    for k in samples[0].keys():
        batch[k] = list()

    for sample in samples:
        for k, v in sample.items():
            batch[k].append(v)

    for k, v in batch.items():
        batch[k] = torch.stack(v)

    return batch