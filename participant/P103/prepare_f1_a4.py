"""核对两份 E4 ZIP 并解出 A4 专用评测副本；不读取评测分数。"""

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS = ROOT / "outputs" / "P103"
DEST = OUTPUTS / "f1-a4-inputs-001"
ZIP_SPECS = {
    "E009": ("e4-freeze-E009-fallback-001", "P103-E009-fallback-E4.zip",
             "cb02973947411d74d56cbfe9ff449d3272775449070c2f7c0fb4d023b8f61cf4"),
    "E015-combo": ("e4-freeze-E015-combo-001", "P103-E015-combo-E4.zip",
                   "d4ef43ec99705e770cb9ef59fdafa345e19081f4de4b829055a075ab02411424"),
}
INPUT_HASHES = {
    "final-confirmation-suite-v1.yaml": "bf3b9ed3084129a59b32c5109216e3e0e3bd09e74a329d29024556e63809d12d",
    "final-confirmation-seeds-v1.json": "4fb1ae3d6376a49bceaefd36c5a2b471623cbbe977ae76167e695a1504ea78ce",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f"评测副本目录已存在，不覆盖：{DEST}")
    protocol = HERE / "F1_A4_PRECOMMITTED_PROTOCOL.md"
    protocol_hash = digest(protocol.read_bytes())
    for name, expected in INPUT_HASHES.items():
        path = HERE / name
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"A4 输入摘要不符：{path}")

    checked = {}
    for role, (folder, archive_name, expected_zip_hash) in ZIP_SPECS.items():
        frozen = OUTPUTS / folder
        archive = frozen / archive_name
        if digest(archive.read_bytes()) != expected_zip_hash:
            raise ValueError(f"冻结 ZIP 摘要不符：{archive}")
        manifest = json.loads((frozen / "freeze-manifest.json").read_text(encoding="utf-8"))
        if manifest["archive_sha256"] != expected_zip_hash:
            raise ValueError(f"冻结清单 ZIP 摘要不符：{frozen}")
        expected_members = {f"P103/{name}" for name in manifest["files"]}
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            if len(members) != len(set(members)) or set(members) != expected_members:
                raise ValueError(f"ZIP 路径与冻结清单不符：{archive}")
            files = {}
            for rel, recorded in manifest["files"].items():
                data = zf.read(f"P103/{rel}")
                actual_hash = digest(data)
                if actual_hash != recorded["sha256"] or len(data) != recorded["size_bytes"]:
                    raise ValueError(f"ZIP 内文件与冻结清单不符：{role}/{rel}")
                files[rel] = data
        checked[role] = files

    DEST.mkdir(parents=True)
    for role, files in checked.items():
        target = DEST / role / "P103"
        for rel, data in files.items():
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    (DEST / "input-manifest.json").write_text(json.dumps({
        "protocol_sha256_before_a4": protocol_hash,
        "suite_sha256": INPUT_HASHES["final-confirmation-suite-v1.yaml"],
        "seeds_sha256": INPUT_HASHES["final-confirmation-seeds-v1.json"],
        "packages": {role: {"archive_sha256": spec[2],
                            "submission": (DEST / role / "P103").relative_to(ROOT).as_posix()}
                     for role, spec in ZIP_SPECS.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已核对并解出两份固定 ZIP；事前协议 SHA-256：{protocol_hash}")


if __name__ == "__main__":
    main()
