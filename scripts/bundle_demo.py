"""Create a checksummed release without raw datasets or training feature caches."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from catalog_search.data import save_json, sha256


ARTIFACTS = ("encoder.joblib", "baseline.txt", "attribute.txt", "no_product_evidence.txt",
             "catalog.parquet", "retrieval.joblib", "index_manifest.json", "metadata.json",
             "protocol.json", "selection.json")
REPORTS = ("summary.json", "metrics.json", "data_audit.json", "latency.json", "results.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True, help="Immutable release-asset URL, not latest/download")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="catalog-bundle-") as temporary:
        staged = Path(temporary)
        for name in ARTIFACTS:
            shutil.copy2(project / "artifacts" / name, staged / name)
        (staged / "reports").mkdir()
        for name in REPORTS:
            shutil.copy2(project / "reports" / name, staged / "reports" / name)
        shutil.copytree(project / "licenses", staged / "licenses")
        shutil.copy2(project / "LICENSE", staged / "licenses/PROJECT-LICENSE")
        shutil.copy2(project / "THIRD_PARTY_NOTICES.md", staged / "THIRD_PARTY_NOTICES.md")
        checksums = {path.relative_to(staged).as_posix(): sha256(path)
                     for path in sorted(staged.rglob("*")) if path.is_file()}
        save_json(staged / "artifact_checksums.json", checksums)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(path.relative_to(staged).as_posix(), date_time=(2026, 10, 26, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    bundle.writestr(info, path.read_bytes())
    manifest = {"version": "0.1.0", "url": args.url, "sha256": sha256(args.output),
                "bytes": args.output.stat().st_size,
                "scope": "Trained models, fitted encoder, 30000-product historical ESCI excerpt, index, reports and licenses; no raw full data or training cache"}
    save_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
