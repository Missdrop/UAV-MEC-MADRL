import os
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "train.ipynb"


class StreamingNotebookClient(NotebookClient):
    """Forward notebook stdout/stderr to the current terminal in real time."""

    def process_message(
        self,
        msg: dict[str, Any],
        cell: NotebookNode,
        cell_index: int,
    ) -> NotebookNode | None:
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def execute_training_notebook() -> None:
    with NOTEBOOK_PATH.open(encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    client = StreamingNotebookClient(
        notebook,
        timeout=None,
        allow_errors=False,
    )
    client.execute()


def main() -> None:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    # Keep train.ipynb's relative output paths rooted in the project directory.
    os.chdir(PROJECT_DIR)
    run_id = 1

    try:
        while True:
            print(f"\n========== Training run {run_id} ==========", flush=True)
            try:
                execute_training_notebook()
            except KeyboardInterrupt:
                raise
            # Keep the unattended loop alive for kernel, cell, and I/O failures.
            except Exception as exc:  # noqa: BLE001
                print(f"\nRun {run_id} failed: {exc}", flush=True)
                time.sleep(3)
            else:
                print(f"\nRun {run_id} completed.", flush=True)
            run_id += 1
    except KeyboardInterrupt:
        print("\nStopped manually.", flush=True)


if __name__ == "__main__":
    main()
