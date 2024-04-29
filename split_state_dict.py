import torch


def split_state_dict(file_path, chunk_size):
    state_dict = torch.load(file_path, map_location="cpu")
    keys = list(state_dict.keys())

    # Iterate over keys and split into chunks
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        chunk_state_dict = {k: state_dict[k] for k in chunk_keys}
        # Save each chunk as a separate file
        torch.save(
            chunk_state_dict,
            f"models/checkpoints/split_state_dict/chunk_{i // chunk_size}.pt",
        )


# Split state dictionary file
split_state_dict(
    "models/checkpoints/python_leq_160_tokens/meta_model_0.pt", chunk_size=50
)
