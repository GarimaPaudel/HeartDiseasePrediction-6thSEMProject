import re
from datetime import datetime
from pathlib import Path

from box import ConfigBox

from src.utils.common import load_json, load_pickle, save_json, save_pickle
from src.utils.logger import logger


class ModelRegistry:
    def __init__(self, config: ConfigBox):
        self.registry_dir = Path(config.registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _next_version(self) -> str:
        existing = [
            d for d in self.registry_dir.iterdir()
            if d.is_dir() and re.match(r"^v\d+$", d.name)
        ]
        if not existing:
            return "v1"
        nums = [int(d.name[1:]) for d in existing]
        return f"v{max(nums) + 1}"

    def register(self, model, metrics: dict, params: dict) -> str:
        version = self._next_version()
        version_dir = self.registry_dir / version
        version_dir.mkdir(parents=True)

        save_pickle(version_dir / "model.pkl", model)
        save_json(version_dir / "metrics.json", metrics)
        save_json(
            version_dir / "metadata.json",
            {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "params": params,
                "sklearn_version": __import__("sklearn").__version__,
            },
        )

        # Update 'latest' symlink
        latest_link = self.registry_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version_dir.resolve())

        logger.info(
            f"Model registered as {version} | "
            f"objective={metrics.get('objective_value'):.4f}"
        )
        return version

    def load_production_model(self):
        latest = self.registry_dir / "latest"
        if not latest.exists():
            raise FileNotFoundError(
                "No production model found. Run the pipeline first."
            )
        model = load_pickle(latest / "model.pkl")
        logger.info(f"Loaded production model from {latest.resolve()}")
        return model

    def list_versions(self) -> list[dict]:
        versions = []
        for d in sorted(self.registry_dir.iterdir()):
            if d.is_dir() and re.match(r"^v\d+$", d.name):
                meta = load_json(d / "metadata.json")
                metrics = load_json(d / "metrics.json")
                versions.append(
                    {
                        "version": d.name,
                        "timestamp": meta.get("timestamp"),
                        "params": meta.get("params"),
                        "accuracy": metrics.get("accuracy"),
                        "sensitivity": metrics.get("sensitivity"),
                        "objective_value": metrics.get("objective_value"),
                    }
                )
        return versions
