"""英雄个人配置：未练跳过 + 优先选择"""
import json
import os
import zhconv

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE = os.path.join(_ROOT, 'hero_config.json')

_config = None


def _load() -> dict:
    global _config
    if _config is None:
        if os.path.exists(_CONFIG_FILE):
            with open(_CONFIG_FILE, encoding='utf-8') as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config


def _n(name: str) -> str:
    return zhconv.convert(name, 'zh-hans')


def is_unpracticed(name: str) -> bool:
    """返回 True 表示该角色在未练名单中，选秀时跳过。"""
    cfg = _load()
    target = _n(name)
    return target in {_n(n) for n in cfg.get('unpracticed', [])}


def is_priority(name: str) -> bool:
    """返回 True 表示该角色在优先名单中，选秀时挪到候选列表最前面。"""
    cfg = _load()
    target = _n(name)
    return target in {_n(n) for n in cfg.get('priority', [])}
