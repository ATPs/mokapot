from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class TempWorkspace:
    """
    Per-run temporary workspace rooted under a user-provided base directory.

    Temporary artifacts are written under:
      <base_dir>/.mokapot-temp/<run_id>/
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.run_id = uuid.uuid4().hex[:12]
        self.path = self.base_dir / ".mokapot-temp" / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def subdir(self, name: str) -> Path:
        out = self.path / name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def cleanup(self):
        temp_root = self.base_dir / ".mokapot-temp"
        try:
            if self.path.exists():
                shutil.rmtree(self.path, ignore_errors=False)
        except Exception as exc:
            LOGGER.warning(
                "Failed to remove temporary workspace '%s': %s",
                self.path,
                exc,
            )
            return

        # Remove the shared temp root when this run was the last workspace.
        try:
            if temp_root.exists():
                temp_root.rmdir()
        except OSError:
            # Parent is not empty (other runs/artifacts still exist).
            pass
        except Exception as exc:
            LOGGER.warning(
                "Failed to remove temporary root '%s': %s",
                temp_root,
                exc,
            )
