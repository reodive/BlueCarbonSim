from typing import Any, Dict


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    超簡易 YAML ローダ（key: value のみ対応、# コメント、空行可）。
    依存を増やさず、基本的な数/文字列/真偽を読み取る。
    """
    cfg: Dict[str, Any] = {}
    try:
        # Read bytes then decode with UTF-8, fallback to cp932 (Windows)
        with open(path, "rb") as fb:
            raw = fb.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp932", errors="ignore")
        for line in text.splitlines(True):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
                k, v = s.split(":", 1)
                key = k.strip()
                val = v.strip()
                # 型推定（真偽/数値/文字列）
                if val.lower() in ("true", "false"):
                    cfg[key] = (val.lower() == "true")
                else:
                    try:
                        if "." in val:
                            cfg[key] = float(val)
                        else:
                            cfg[key] = int(val)
                    except ValueError:
                        cfg[key] = val
    except FileNotFoundError:
        pass
    return cfg

