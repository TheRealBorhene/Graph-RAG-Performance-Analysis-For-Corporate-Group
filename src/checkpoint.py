import os
import json

CHECKPOINT_PATH = "../data/extractor_checkpoint.json"


def load_checkpoint() -> dict:
    # if a checkpoint file exists, load it and resume from where we stopped
    # if not, return an empty state (fresh start)
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        last = checkpoint.get("last_chunk_id", -1)
        print(f"Checkpoint found — resuming from chunk {last + 1}")
        print(f"Already extracted: {len(checkpoint['entities'])} entities, {len(checkpoint['relationships'])} relationships\n")
        return checkpoint
    else:
        print("No checkpoint found — starting fresh\n")
        return {
            "last_chunk_id": -1,
            "entities":      [],
            "relationships": []
        }


def save_checkpoint(last_chunk_id: int, entities: list, relationships: list) -> None:
    # save after every single chunk
    # if the script stops, we lose at most one chunk of work
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "last_chunk_id": last_chunk_id,
            "entities":      entities,
            "relationships": relationships
        }, f, indent=2, ensure_ascii=False)


def delete_checkpoint() -> None:
    # called when extraction is fully complete
    # deletes the checkpoint so the next run starts fresh
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint deleted — extraction complete")


def is_already_processed(chunk_id: int, last_chunk_id: int) -> bool:
    # returns True if this chunk was already processed in a previous run
    return chunk_id <= last_chunk_id