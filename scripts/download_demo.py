"""Download the project's published demo bundle and verify every extracted file.

Only load joblib artifacts from a source you trust: joblib is Python pickle.
The checked-in release manifest pins the author's own bundle by SHA256.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile


def install(manifest, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    with tempfile.TemporaryDirectory(prefix="catalog-download-") as temporary:
        archive = Path(temporary) / "demo.zip"
        request = urllib.request.Request(manifest["url"], headers={"User-Agent": "catalog-search/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        actual = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        if actual != manifest["sha256"]:
            raise ValueError("Release checksum mismatch; nothing installed")
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                    raise ValueError("Unsafe archive path")
                if item.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError("Symlinks are not supported")
            staged = Path(temporary) / "unpacked"
            bundle.extractall(staged)
        checksums = json.loads((staged / "artifact_checksums.json").read_text())
        for name, expected in checksums.items():
            path = staged / name
            if not path.is_file() or hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, output, dirs_exist_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("demo_release.json"))
    args = parser.parse_args()
    result = install(json.loads(args.manifest.read_text()), args.output)
    print(f"Verified demo installed: {result}")


if __name__ == "__main__":
    main()
