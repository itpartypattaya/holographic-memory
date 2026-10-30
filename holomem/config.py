"""Settings of the bridge: ``$HOLOMEM_HOME/config.toml`` (``~/.holographic/config.toml`` by default).

Every key is optional. A value that is not of its kind, outside its set, or a file that does not parse turns memory off
(``problems``): the hooks then recall nothing and the tools say why. The reason names the key and what it must be,
never the value: a key or token pasted into the wrong line must not reach a prompt or a log. A key the bridge does
not know is a warning (``warnings``): a typo would otherwise leave its setting at the default without a word.
``rerank_url`` holding credentials (``user:pass@``), a query or a fragment is a problem too: the engine would keep them
in the address it calls, and the bridge never stores or shows an address.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tomllib
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from .engine.home import ENV_HOME, home  # noqa: F401 - ENV_HOME re-exported for callers of the settings

# key -> (default, kind, allowed): kind float/int/bool/str/text/path/names; allowed = a set of strings or (low, high)
# for numbers; str is a choice (lower-cased), text is kept as written (an address, a model name), names a list of them
_KEYS: dict[str, tuple] = {
    "db_path": ("", "path", None),
    "scope": ("global", "str", {"global", "project"}),
    "default_trust": (0.5, "float", (0.0, 1.0)),
    "min_trust_threshold": (0.3, "float", (0.0, 1.0)),
    "hrr_dim": (1024, "int", (16, 16384)),
    "hrr_weight": (0.3, "float", (0.0, 1.0)),
    "temporal_decay_half_life": (0, "int", (0, 100000)),
    "prefetch_min_coverage": (0.3, "float", (0.0, 1.0)),
    "prefetch_repeat_turns": (6, "int", (0, 10000)),
    "prefetch_max_tokens": (0, "int", (0, 1000000)),
    "prefetch_max_facts": (5, "int", (1, 20)),
    "recall_source": ("off", "str", {"off", "short"}),
    "auto_extract": (False, "bool", None),
    "store_policy": ("ask", "str", {"ask", "proactive"}),
    "hook_deadline_seconds": (8.0, "float", (0.5, 25.0)),
    # the names of the person the memory belongs to: named in passing all the time, they pull no facts by name alone
    "self_names": ((), "names", None),
    # the task-outcome journal (journal.py): receipts of what recall and search offered, outcomes the agent reports
    "experience": (False, "bool", None),
    "experience_trace": (False, "bool", None),
    # search by meaning (ONLINE when on): the embedding_* keys, read by the engine (embeddings.Settings)
    "embeddings": (False, "bool", None),
    "embedding_url": ("https://api.openai.com/v1", "text", None),
    "embedding_model": ("text-embedding-3-small", "text", None),
    "embedding_key_env": ("OPENAI_API_KEY", "text", None),
    "embedding_dimensions": (None, "int", (1, 65536)),
    "embedding_timeout": (3.0, "float", (0.5, 6.0)),
    "embedding_min_score": (None, "float", (-1.0, 1.0)),
    "embedding_prefetch_facts": (1, "int", (0, 3)),
    "embedding_weight": (0.3, "float", (0.0, 0.8)),
    "embedding_space": ("", "text", None),
    # rescoring by a cross-encoder (rescore.py): the rerank_* keys, read by the engine (rerank.Settings)
    "rerank": (False, "bool", None),
    "rerank_url": ("http://127.0.0.1:8080/v1/rerank", "text", None),
    "rerank_model": ("local-reranker", "text", None),
    "rerank_top_k": (15, "int", (1, 1000)),  # 40 took ~4.6 s through a tunnel to a 2-thread CPU server
    "rerank_weight": (0.0, "float", (0.0, 1.0)),
    "rerank_window": (0.05, "float", (0.0, 1.0)),
    "rerank_scale": ("logit", "str", {"logit", "unit"}),
    "rerank_timeout": (3.0, "float", (0.5, 10.0)),
}


@dataclass
class Config:
    home: Path
    db_path: Path
    scope: str = "global"
    default_trust: float = 0.5
    min_trust_threshold: float = 0.3
    hrr_dim: int = 1024
    hrr_weight: float = 0.3
    temporal_decay_half_life: int = 0
    prefetch_min_coverage: float = 0.3
    prefetch_repeat_turns: int = 6
    prefetch_max_tokens: int = 0
    prefetch_max_facts: int = 5
    recall_source: str = "off"
    auto_extract: bool = False
    store_policy: str = "ask"
    hook_deadline_seconds: float = 8.0
    self_names: tuple = ()
    experience: bool = False
    experience_trace: bool = False
    embeddings: bool = False
    embedding_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_key_env: str = "OPENAI_API_KEY"
    embedding_dimensions: int | None = None
    embedding_timeout: float = 3.0
    embedding_min_score: float | None = None
    embedding_prefetch_facts: int = 1
    embedding_weight: float = 0.3
    embedding_space: str = ""
    rerank: bool = False
    rerank_url: str = "http://127.0.0.1:8080/v1/rerank"
    rerank_model: str = "local-reranker"
    rerank_top_k: int = 15
    rerank_weight: float = 0.0
    rerank_window: float = 0.05
    rerank_scale: str = "logit"
    rerank_timeout: float = 3.0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems

    @property
    def state_path(self) -> Path:
        """The bridge's own bookkeeping (turn numbers, what was recalled when): never inside the fact store."""
        return self.home / "state.db"

    @property
    def secrets_path(self) -> Path:
        """KEY=VALUE lines for the embeddings key: read before the environment, so the key need not be a variable of the
        whole user session (where every other program that knows the name would pick it up)."""
        return self.home / ".env"

    def embedding_config(self) -> dict:
        """The config keys of search by meaning, as the engine's embeddings.settings_from reads them."""
        keys = {key: getattr(self, key) for key in _KEYS if key.startswith("embedding")}
        return {k: v for k, v in keys.items() if v is not None and v != ""}

    def rerank_config(self) -> dict:
        """The config keys of rescoring, as the engine's rerank.settings_from reads them (with the embeddings
        switch and timeout: the engine keeps the two calls of one turn within its 7.5 s together)."""
        keys = {key: getattr(self, key) for key in _KEYS if key.startswith("rerank")}
        return {**keys, "embeddings": self.embeddings, "embedding_timeout": self.embedding_timeout}

    def database_for(self, cwd: str | Path | None) -> Path | None:
        """The fact store for a session started in *cwd*: the one database in global scope; in project scope one per
        project (the git root above *cwd*, else *cwd* itself), None without a directory."""
        if self.scope == "global":
            return self.db_path
        if not cwd:
            return None
        root = project_root(Path(cwd))
        return self.home / "projects" / f"{project_key(root)}.db"


def project_root(path: Path) -> Path:
    try:
        path = path.expanduser().resolve()
    except OSError:
        path = path.expanduser().absolute()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def project_key(root: Path) -> str:
    """A file name for a project: its folder name made safe, cut to 40 characters, plus 10 hex of the full path's
    sha256 (the same folder name in two places is two projects). Case-folded on Windows, where paths are."""
    text = str(root)
    if sys.platform == "win32":
        text = text.lower()
    name = _SAFE.sub("_", root.name or "root")[:40]
    return f"{name}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _coerce(key: str, value, kind: str, allowed):
    """(value, problem): a problem names the key and the expected kind, never the value."""
    if kind == "bool":
        if isinstance(value, bool):
            return value, None
        return None, f"{key} must be true or false"
    if kind in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind == "int" and isinstance(value, float)
                                                                               and not value.is_integer()):
            return None, f"{key} must be a{'n integer' if kind == 'int' else ' number'}"
        number = int(value) if kind == "int" else float(value)
        low, high = allowed
        if not low <= number <= high:
            return None, f"{key} must be between {low} and {high}"
        return number, None
    if kind == "names":
        if not isinstance(value, list) or not all(isinstance(v, str) and 0 < len(v.strip()) <= 80 for v in value):
            return None, f"{key} must be a list of names"
        return tuple(v.strip() for v in value), None
    if not isinstance(value, str):
        return None, f"{key} must be a string"
    if kind == "path":
        return value, None
    if kind == "text":
        return value.strip(), None
    text = value.strip().lower()
    if allowed and text not in allowed:
        return None, f"{key} must be one of: {', '.join(sorted(allowed))}"
    return text, None


def _url_problem(key: str, value: str) -> str | None:
    """A rerank_url that carries credentials, a query or a fragment: named by the key, never shown."""
    try:
        parts = urllib.parse.urlsplit(value)
        userinfo = "@" in parts.netloc or parts.username is not None or parts.password is not None
    except ValueError:
        return f"{key} is not a valid address"
    if userinfo or parts.query or parts.fragment or "?" in value or "#" in value:
        return f"{key} must not carry credentials or a query"
    return None


_CHECKS = {"rerank_url": _url_problem}


def load(path: Path | None = None) -> Config:
    base = home()
    path = path or base / "config.toml"
    values: dict = {}
    problems: list[str] = []
    warnings: list[str] = []
    if path.exists():
        try:
            values = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            # The parser's message quotes the line it stopped at, which may hold a secret: the kind only
            problems.append(f"{path.name} could not be read ({type(exc).__name__}); fix the file or remove it")
            values = {}
    settings: dict = {}
    for key, value in values.items():
        if key not in _KEYS:
            warnings.append(f"{key} is not a setting of holomem; ignored (a typo?)")
            continue
        default, kind, allowed = _KEYS[key]
        coerced, problem = _coerce(key, value, kind, allowed)
        if not problem and key in _CHECKS:
            problem = _CHECKS[key](key, coerced)
        if problem:
            problems.append(problem)
        else:
            settings[key] = coerced
    raw_db = settings.pop("db_path", "") or ""
    db_path = _expand(raw_db) if raw_db else base / "memory_store.db"
    return Config(home=base, db_path=db_path, problems=problems, warnings=warnings, **settings)
