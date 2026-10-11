"""Configuration wrapper with typed accessors, presets, prompt history, and export settings."""
import yaml
import os
import copy
import logging
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".aimusic")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {
        "cache_dir": os.path.join(CONFIG_DIR, "cache"),
        "ckpt_dir": os.path.join(CONFIG_DIR, "ckpt"),
        "heartlib_repo_dir": "",
    },
    "hardware": {"device": "auto"},
    "plugins": {"active_plugin": ""},
    "settings": {
        "output_dir": os.path.join(os.path.expanduser("~"), "aimusic_output"),
        "export_format": "WAV",
        "auto_play": True,
        "organize_output": True,
        "window_geometry": "",
        "auto_reveal": False,
    },
    "post_processing": {"normalize": True, "fade_ms": 50},
    "presets": {},
    "prompt_history": [],
    "backend_settings": {},
}

MAX_HISTORY = 50

class Config:
    def __init__(self, data: dict):
        self._data = data

    @property
    def raw(self) -> dict:
        return self._data

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def cache_dir(self) -> str:
        return os.path.expanduser(self._data.get("system", {}).get("cache_dir", ""))

    @property
    def ckpt_dir(self) -> str:
        return os.path.expanduser(self._data.get("system", {}).get("ckpt_dir", ""))

    @property
    def heartlib_repo_dir(self) -> str:
        return os.path.expanduser(self._data.get("system", {}).get("heartlib_repo_dir", ""))

    def set_system_path(self, key: str, value: str) -> None:
        self._data.setdefault("system", {})[key] = value

    @property
    def device(self) -> str:
        return self._data.get("hardware", {}).get("device", "auto")

    @property
    def hardware_config(self) -> dict:
        return self._data.get("hardware", {})

    @property
    def active_plugin(self) -> str:
        return self._data.get("plugins", {}).get("active_plugin", "")

    def set_active_plugin(self, name: str) -> None:
        self._data.setdefault("plugins", {})["active_plugin"] = name

    @property
    def output_dir(self) -> str:
        return os.path.expanduser(self._data.get("settings", {}).get("output_dir", "output"))

    def set_output_dir(self, path: str) -> None:
        self._data.setdefault("settings", {})["output_dir"] = path

    @property
    def export_format(self) -> str:
        return self._data.get("settings", {}).get("export_format", "WAV")

    def set_export_format(self, fmt: str) -> None:
        self._data.setdefault("settings", {})["export_format"] = fmt

    @property
    def auto_play(self) -> bool:
        return self._data.get("settings", {}).get("auto_play", True)

    def set_auto_play(self, val: bool) -> None:
        self._data.setdefault("settings", {})["auto_play"] = val

    @property
    def organize_output(self) -> bool:
        return self._data.get("settings", {}).get("organize_output", True)

    def set_organize_output(self, val: bool) -> None:
        self._data.setdefault("settings", {})["organize_output"] = val

    @property
    def post_processing(self) -> dict:
        return self._data.get("post_processing", {"normalize": True, "fade_ms": 50})

    def set_post_processing(self, key: str, value: Any) -> None:
        self._data.setdefault("post_processing", {})[key] = value

    def get_backend_settings(self, backend_name: str) -> dict:
        return copy.deepcopy(self._data.get("backend_settings", {}).get(backend_name, {}))

    def update_backend_setting(self, backend_name: str, key: str, value: Any) -> None:
        self._data.setdefault("backend_settings", {}).setdefault(backend_name, {})
        self._data["backend_settings"][backend_name][key] = value

    @property
    def presets(self) -> dict:
        return self._data.get("presets", {})

    def get_presets_for_backend(self, backend_name: str) -> dict:
        return self._data.get("presets", {}).get(backend_name, {})

    def save_preset(self, backend_name: str, preset_name: str, settings: dict) -> None:
        self._data.setdefault("presets", {}).setdefault(backend_name, {})
        self._data["presets"][backend_name][preset_name] = settings

    def delete_preset(self, backend_name: str, preset_name: str) -> None:
        presets = self._data.get("presets", {}).get(backend_name, {})
        presets.pop(preset_name, None)

    @property
    def prompt_history(self) -> List[str]:
        return self._data.get("prompt_history", [])

    def add_prompt_to_history(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        history = self._data.setdefault("prompt_history", [])
        if prompt in history:
            history.remove(prompt)
        history.insert(0, prompt)
        del history[MAX_HISTORY:]

    def clear_prompt_history(self) -> None:
        self._data["prompt_history"] = []

    def save(self, path: Optional[str] = None) -> None:
        path = path or DEFAULT_CONFIG_PATH
        dir_name = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_name, exist_ok=True)

        data = yaml.dump(self._data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix=".cfg_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.warning("Atomic save failed, falling back to direct write")
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        path = path or DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            cfg = cls(copy.deepcopy(DEFAULT_CONFIG))
            cfg.save(path)
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return cls(copy.deepcopy(DEFAULT_CONFIG))
            merged = copy.deepcopy(DEFAULT_CONFIG)
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
            return cls(merged)
        except Exception as exc:
            logger.error("Config load error: %s — using defaults", exc)
            return cls(copy.deepcopy(DEFAULT_CONFIG))