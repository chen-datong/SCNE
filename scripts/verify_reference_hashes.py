"""Verify that manuscript reference PDFs have not changed."""

from hashlib import sha256
from pathlib import Path


EXPECTED = {
    "SE_vs_NM_and_n.pdf": "16815bf1a2733e65129f7b9a2c6d1cec061258d413b3e31578cd71331d5c8911",
    "vcooling.pdf": "8679159296ca91bdfa0cb31ff3d74f25a814ac8738e4f268c0d2944fabdac448",
    "PTME.pdf": "898b0e00b02be64f6fb421588c47d5f6735f65919dc67719e82a86055e8e3c1d",
    "sample_lower.pdf": "69c82583e01584941da11020a8f5e8f7a0112dd4af1f9595fdb82c286b70f18c",
    "SE_all_observables.pdf": "f468a4a56a4b9a6488c40d4a750b89ef2726cfadd04b36f6e75adcd1e372679e",
    "SE_vs_NM_and_nA.pdf": "a26a359eaa8fa6b7ead12bcd55a8d31f6445f437042dfaa1f97a2addccf74080",
    "Gibbs_ent_detect.pdf": "a9929c25eba89127f7d968927e3b054ae271feb7b5a74a2db6bf1262ba93e514",
    "Depol_ent_detect.pdf": "6c34e58dfa4a193a98a6189a7b0857cb52a7dde90eb21a5ac253ce10a930d97d",
}


def main():
    root = Path(__file__).resolve().parents[1] / "reference" / "paper_figures"
    failed = False
    for name, expected in EXPECTED.items():
        path = root / name
        actual = sha256(path.read_bytes()).hexdigest()
        ok = actual == expected
        print(f"{'OK' if ok else 'FAIL'}  {name}  {actual}")
        failed |= not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
