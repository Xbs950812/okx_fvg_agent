"""
打包工具 — 更新代码后打包 zip，手动发给买家

用法:
  python delivery.py pack             # 打包源码为 zip（不含敏感文件）
"""

import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
PACK_DIR = BASE_DIR / "packages"
CONFIG_PATH = BASE_DIR / "config.json"

# 默认夸克网盘链接（config.json 可覆盖）
DEFAULT_QUARK_URL = "https://pan.quark.cn/s/8320adb53d0b"


def _load_quark_url() -> str:
    """从 config.json 读取夸克网盘链接，若不可用则返回默认值。"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("delivery", {}).get("quark_url", DEFAULT_QUARK_URL)
    except Exception:
        pass
    return DEFAULT_QUARK_URL

EXCLUDE = {
    "config.json", "orders.json", "agent_state.json", "agent.log",
    "delivery.py", "pay_server.py",
    "__pycache__", ".git", "packages", "reports", "memory", "debate_checkpoints",
    # 修复 Bug 46: 排除 IDE/编辑器/虚拟环境目录，避免泄露开发环境
    ".vscode", ".idea", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env", "node_modules",
    # 排除 OS 垃圾文件
    ".DS_Store", "Thumbs.db", "desktop.ini",
}
EXCLUDE_EXT = {".pyc", ".pyo", ".log", ".swp", ".swo", ".bak", ".tmp"}


def pack():
    PACK_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d")
    zip_path = PACK_DIR / f"okx_fvg_agent_{version}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE]
            for f in files:
                if f in EXCLUDE or os.path.splitext(f)[1].lower() in EXCLUDE_EXT:
                    continue
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(BASE_DIR))

    size_mb = zip_path.stat().st_size / (1024*1024)
    quark_url = _load_quark_url()
    print(f"\n  打包完成: {zip_path} ({size_mb:.2f} MB)")
    print(f"  夸克网盘: {quark_url}")


if __name__ == "__main__":
    pack()