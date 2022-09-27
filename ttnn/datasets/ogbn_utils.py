from ogb.nodeproppred import NodePropPredDataset


def load_ogb_fixed_split_dataset(name, data_path):
    """Load an OGB dataset with a fixed train/val/test split."""
    dataset = NodePropPredDataset(name=name, root=data_path)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"], split_idx["valid"], split_idx["test"])
    fixed_split_indices = [train_idx, valid_idx, test_idx]
    graph, label = dataset[0]  # graph: library-agnostic graph object
    return graph, label, fixed_split_indices
