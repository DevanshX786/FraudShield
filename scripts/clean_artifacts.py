"""Clean generated local artifacts and test caches."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    print("=== Starting FraudShield Artifacts Cleanup ===")
    project_root = Path(__file__).resolve().parent.parent

    # 1. Clean directories that should be entirely removed
    dirs_to_remove = [
        project_root / ".pytest_cache",
        project_root / ".coverage",
        project_root / "htmlcov",
    ]
    for d in dirs_to_remove:
        if d.exists():
            try:
                if d.is_dir():
                    shutil.rmtree(d)
                else:
                    d.unlink()
                print(f"Removed: {d.relative_to(project_root)}")
            except Exception as e:
                print(f"Failed to remove {d}: {e}")

    # 2. Recursively remove __pycache__ directories
    for p in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(p)
            print(f"Removed cache directory: {p.relative_to(project_root)}")
        except Exception as e:
            print(f"Failed to remove cache directory {p}: {e}")

    # 3. Clean files in directories but keep .gitkeep files
    dirs_to_clean = [
        project_root / "logs",
        project_root / "models",
        project_root / "reports",
        project_root / "data" / "processed",
        project_root / "data" / "inference_batches",
    ]

    for d in dirs_to_clean:
        if d.exists() and d.is_dir():
            for item in d.iterdir():
                if item.name == ".gitkeep":
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print(f"Cleaned: {item.relative_to(project_root)}")
                except Exception as e:
                    print(f"Failed to clean item {item}: {e}")

    # 4. Remove root-level SQLite databases (*.db)
    for db_file in project_root.glob("*.db"):
        try:
            db_file.unlink()
            print(f"Removed database file: {db_file.name}")
        except Exception as e:
            print(f"Failed to remove database file {db_file}: {e}")

    print("=== Cleanup Completed Successfully ===")


if __name__ == "__main__":
    main()
