"""
config.py — Configuration loader for both central server and edge nodes.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("[Config] ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)


class _Namespace:
    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


_REQUIRED_EDGE = [
    "node.id", "node.role",
    "central.host", "central.port",
    "model.path", "model.imgsz",
    "pipeline.frame_height", "pipeline.frame_width", "pipeline.frame_channels",
    "decoder.pace_min", "decoder.pace_max",
    "reporter.poll_interval_s", "reporter.heartbeat_interval_s",
]

_REQUIRED_CENTRAL = [
    "node.id", "node.role",
    "server.host", "server.port",
    "heartbeat.timeout_s",
]


def _get_nested(data: dict, key: str) -> Any:
    for k in key.split("."):
        if not isinstance(data, dict) or k not in data:
            return None
        data = data[k]
    return data


def _validate(data: dict, keys: list, path: str):
    missing = [k for k in keys if _get_nested(data, k) is None]
    if missing:
        print(f"[Config] ERROR: Missing required fields in {path}:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


class Config:
    _instance = None

    def __init__(self, data: dict, path: str):
        self._data = data
        self._path = path
        self._ns   = _Namespace(data)

    def __getattr__(self, name: str):
        try:
            return getattr(self._ns, name)
        except AttributeError:
            raise AttributeError(f"[Config] No key '{name}' in {self._path}")

    def get(self, key: str, default: Any = None) -> Any:
        v = _get_nested(self._data, key)
        return v if v is not None else default

    def role(self) -> str:
        return self._data.get("node", {}).get("role", "edge")

    def is_central(self) -> bool:
        return self.role() == "central"

    def is_edge(self) -> bool:
        return self.role() == "edge"

    def to_dict(self) -> dict:
        """Return a deep copy safe for pickling across process boundaries."""
        import copy
        return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Reconstruct from a dict — zero disk I/O. Used in subprocesses."""
        inst = cls(data, path="<passed-from-parent>")
        cls._instance = inst
        return inst

    @classmethod
    def load(cls, path: str = None) -> "Config":
        if path is None:
            path = os.environ.get("CCTV_CONFIG")
        if path is None:
            candidates = [
                Path("config.yaml"),
                Path("../config.yaml"),
                Path(__file__).parent.parent / "config.yaml",
                Path(__file__).parent / "config.yaml",
            ]
            for c in candidates:
                if c.exists():
                    path = str(c)
                    break
        if path is None:
            print("[Config] ERROR: No config.yaml found.")
            print("  Place config.yaml next to your script, or set CCTV_CONFIG env var.")
            sys.exit(1)

        path = str(Path(path).resolve())

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Config] ERROR: File not found: {path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"[Config] ERROR: Invalid YAML: {e}")
            sys.exit(1)

        if not isinstance(data, dict):
            print(f"[Config] ERROR: config.yaml must be a dict, got {type(data)}")
            sys.exit(1)

        role = data.get("node", {}).get("role", "edge")
        _validate(data, _REQUIRED_CENTRAL if role == "central" else _REQUIRED_EDGE, path)

        inst = cls(data, path)
        cls._instance = inst
        print(f"[Config] Loaded {path} (role={role}, node_id={data.get('node',{}).get('id','?')})")
        return inst


def get_config(path: str = None) -> Config:
    """Return singleton config, loading if needed."""
    if Config._instance is None:
        Config.load(path)
    return Config._instance


def setup_logging(cfg: Config):
    level    = getattr(logging, cfg.get("logging.level", "INFO").upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    lf = cfg.get("logging.file")
    if lf:
        p = Path(lf)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Auto-load on import — ONLY in the main process, never in spawned subprocesses.
# Subprocesses receive cfg_dict from the parent and call Config.from_dict()
# so they must NOT trigger a disk read here.
# Detection: multiprocessing.current_process().name is always 'MainProcess'
# in the parent; spawned workers get names like 'inference', 'decoder-0', etc.
cfg: Config = None

def _auto_load():
    global cfg
    import multiprocessing as _mp
    if _mp.current_process().name != 'MainProcess':
        return   # subprocess — skip disk read
    candidates = [
        Path("config.yaml"),
        Path("../config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            cfg = Config.load(str(c))
            return

_auto_load()