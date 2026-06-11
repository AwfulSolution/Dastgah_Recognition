import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dastgah_v2.data import build_manifest, load_splits  # noqa: E402


EXTERNAL_ORDER = ["Segah", "RastPanjgah", "Nava", "Mahour", "Homayun", "Chahargah", "Shur"]
OURS_TO_EXTERNAL = {
    "Chahargah": "Chahargah",
    "Homayun": "Homayun",
    "Mahur": "Mahour",
    "Nava": "Nava",
    "Segah": "Segah",
    "Shur": "Shur",
}


@dataclass
class SampleRef:
    source_path: Path
    our_label: str
    ext_label: str
    stem: str
    csv_path: Path


def _run(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)} (cwd={cwd})\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _check_essentia() -> None:
    try:
        import essentia  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Essentia is required for luciamarock pipeline and is not available in current environment."
        ) from exc


def _patch_external_compat(work_repo: Path) -> None:
    tonicext = work_repo / "pdct_ir" / "tonicext.py"
    txt = tonicext.read_text(encoding="utf-8")
    txt = txt.replace("for i in range (ccount/2):", "for i in range (ccount//2):")
    tonicext.write_text(txt, encoding="utf-8")

    # Python 3 rejects the single tab-indented line in this file (TabError).
    melody = work_repo / "pdct_ir" / "predominantMelodyMakam.py"
    txt = melody.read_text(encoding="utf-8")
    txt = txt.replace("\tfor r in rmv_idx:", "        for r in rmv_idx:")
    melody.write_text(txt, encoding="utf-8")

    # numpy >= 1.22 raises TypeError (not ValueError) when np.loadtxt gets an
    # in-memory array instead of a path.
    pdist = work_repo / "pdct_ir" / "pitchDistribution.py"
    txt = pdist.read_text(encoding="utf-8")
    txt = re.sub(
        r"(np\.loadtxt\([^)]*\)\n\s+except )ValueError:",
        r"\1(TypeError, ValueError):",
        txt,
    )
    pdist.write_text(txt, encoding="utf-8")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _prepare_worktree(external_repo: Path, run_dir: Path, resume: bool = False) -> Path:
    work_repo = run_dir / "work_external"
    if work_repo.exists():
        if resume:
            _patch_external_compat(work_repo)
            return work_repo
        shutil.rmtree(work_repo)
    shutil.copytree(external_repo, work_repo)
    _patch_external_compat(work_repo)

    data_dir = work_repo / "pdct_ir" / "data"
    for p in data_dir.glob("*"):
        if p.name == "annotations.json":
            continue
        if p.is_file():
            p.unlink()

    scores_dir = work_repo / "markov" / "scores"
    for p in scores_dir.glob("*"):
        if p.is_file():
            p.unlink()

    matrices_dir = work_repo / "markov" / "matrices"
    for p in matrices_dir.glob("*"):
        if p.name in {"read_matrix.py", "expected.py", "statistics.py", "nrows.sh"}:
            continue
        if p.is_file():
            p.unlink()

    for p in (work_repo / "markov").glob("*.csv"):
        if p.is_file():
            p.unlink()

    for name in ("results.log", "expected.log", "scores.res"):
        _safe_unlink(matrices_dir / name)
    return work_repo


def _load_split_tracks(data_root: Path, splits_path: Path) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    tracks = build_manifest(str(data_root))
    splits = load_splits(str(splits_path))

    def pick(idxs: List[int]) -> List[Tuple[Path, str]]:
        out: List[Tuple[Path, str]] = []
        for i in idxs:
            t = tracks[i]
            out.append((Path(t.path), t.label))
        return out

    train = pick(splits["train"])
    test = pick(splits["test"])
    return train, test


def _convert_to_mp3_wav(src_audio: Path, dst_mp3: Path, dst_wav: Path, sample_rate: int = 44100) -> None:
    if src_audio.suffix.lower() != ".mp3":
        raise ValueError(
            f"External pipeline expects MP3 inputs. Found non-MP3 file: {src_audio}. "
            "Convert dataset to MP3 first."
        )
    audio, _ = librosa.load(str(src_audio), sr=sample_rate, mono=True)
    sf.write(str(dst_wav), audio, sample_rate)
    shutil.copy2(src_audio, dst_mp3)


def _csv_is_usable(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
    except OSError:
        return False
    return len(first.split("\t")) >= 2 and len(second.split("\t")) >= 2


def _process_sample(work_repo: Path, ref: SampleRef) -> None:
    """Convert one source track and run the external pitch pipeline on it.

    Staged wav/mp3 and intermediate pitch files are deleted right after so
    disk usage stays bounded; only the final per-track csv is kept.
    """
    pdct_dir = work_repo / "pdct_ir"
    data_dir = pdct_dir / "data"
    dst_mp3 = data_dir / f"{ref.stem}.mp3"
    dst_wav = data_dir / f"{ref.stem}.wav"
    try:
        _convert_to_mp3_wav(ref.source_path, dst_mp3, dst_wav)
        _run([sys.executable, "main.py", ref.stem], cwd=pdct_dir)
    finally:
        for name in (
            f"{ref.stem}.mp3",
            f"{ref.stem}.wav",
            f"{ref.stem}.pitch",
            f"{ref.stem}.pitch_hist_wrtTonic.json",
        ):
            _safe_unlink(data_dir / name)


def _extract_all(
    work_repo: Path,
    refs: List[SampleRef],
    num_workers: int,
    desc: str,
) -> Tuple[List[SampleRef], List[SampleRef]]:
    pending = [r for r in refs if not r.csv_path.exists()]
    cached = len(refs) - len(pending)
    if cached:
        print(f"{desc}: {cached} file(s) already extracted, skipping")

    failed: List[SampleRef] = []

    def warn(ref: SampleRef, exc: Exception) -> None:
        print(f"WARNING: pitch extraction failed for {ref.source_path.name}: {exc}")

    if num_workers <= 1:
        for ref in tqdm(pending, desc=desc):
            try:
                _process_sample(work_repo, ref)
            except Exception as exc:
                warn(ref, exc)
                failed.append(ref)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_process_sample, work_repo, ref): ref for ref in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                ref = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    warn(ref, exc)
                    failed.append(ref)

    ok = [r for r in refs if r not in failed and r.csv_path.exists()]
    return ok, failed


def _stage_samples(
    work_repo: Path,
    train: List[Tuple[Path, str]],
    test: List[Tuple[Path, str]],
    num_workers: int,
) -> Tuple[Dict[str, List[SampleRef]], List[SampleRef], int]:
    data_dir = work_repo / "pdct_ir" / "data"

    counters_train: Dict[str, int] = defaultdict(int)
    counters_test: Dict[str, int] = defaultdict(int)
    train_flat: List[SampleRef] = []
    test_refs: List[SampleRef] = []

    for src, our_label in train:
        ext_label = OURS_TO_EXTERNAL[our_label]
        counters_train[ext_label] += 1
        stem = f"{ext_label}_{counters_train[ext_label]}"
        train_flat.append(
            SampleRef(source_path=src, our_label=our_label, ext_label=ext_label, stem=stem, csv_path=data_dir / f"{stem}.csv")
        )

    for src, our_label in test:
        ext_label = OURS_TO_EXTERNAL[our_label]
        counters_test[ext_label] += 1
        stem = f"{ext_label}_test_{counters_test[ext_label]}"
        test_refs.append(
            SampleRef(source_path=src, our_label=our_label, ext_label=ext_label, stem=stem, csv_path=data_dir / f"{stem}.csv")
        )

    train_ok, train_failed = _extract_all(work_repo, train_flat, num_workers, "Pitch extract train")
    test_ok, test_failed = _extract_all(work_repo, test_refs, num_workers, "Pitch extract test")

    train_refs: Dict[str, List[SampleRef]] = defaultdict(list)
    for ref in train_ok:
        train_refs[ref.ext_label].append(ref)

    return train_refs, test_ok, len(train_failed) + len(test_failed)


def _build_class_matrices(work_repo: Path, train_refs: Dict[str, List[SampleRef]]) -> None:
    markov_dir = work_repo / "markov"
    matrices_dir = markov_dir / "matrices"

    for ext_label, refs in train_refs.items():
        usable = [r for r in refs if _csv_is_usable(r.csv_path)]
        skipped = len(refs) - len(usable)
        if skipped:
            print(f"WARNING: skipped {skipped} unusable train csv(s) for {ext_label}")
        if not usable:
            continue
        # read_float_notet.py expects <label>_1.csv .. <label>_n.csv in its cwd
        # (upstream processing.sh copies them from scores/), so stage them with
        # contiguous numbering here.
        staged: List[Path] = []
        for j, ref in enumerate(usable, start=1):
            dst = markov_dir / f"{ext_label}_{j}.csv"
            shutil.copy2(ref.csv_path, dst)
            staged.append(dst)
        _run([sys.executable, "read_float_notet.py", ext_label, str(len(usable))], cwd=markov_dir)
        for p in staged:
            _safe_unlink(p)
        src = markov_dir / f"{ext_label}_matrix.csv"
        if not src.exists():
            raise FileNotFoundError(f"Missing generated matrix: {src}")
        shutil.move(str(src), str(matrices_dir / f"{ext_label}_matrix.csv"))

    # The upstream read_matrix.py always expects RastPanjgah matrix.
    rast_matrix = matrices_dir / "RastPanjgah_matrix.csv"
    if not rast_matrix.exists():
        # Create a neutral zero matrix if this class is absent in dataset.
        template = matrices_dir / "Segah_matrix.csv"
        mat = np.genfromtxt(str(template))
        zeros = np.zeros_like(mat)
        np.savetxt(str(rast_matrix), zeros)


def _predict_test(work_repo: Path, test_refs: List[SampleRef]) -> Tuple[List[int], List[int], List[Dict[str, str]]]:
    markov_dir = work_repo / "markov"
    matrices_dir = markov_dir / "matrices"
    results_log = matrices_dir / "results.log"
    expected_log = matrices_dir / "expected.log"
    _safe_unlink(results_log)
    _safe_unlink(expected_log)
    results_log.touch()
    expected_log.touch()

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict[str, str]] = []

    for ref in tqdm(test_refs, desc="Markov classify test"):
        if not _csv_is_usable(ref.csv_path):
            print(f"WARNING: skipped unusable test csv for {ref.source_path.name}")
            rows.append(
                {
                    "file": ref.source_path.name,
                    "true_label": ref.ext_label,
                    "pred_label": "SKIPPED",
                    "pred_idx": "",
                }
            )
            continue

        shutil.copy2(ref.csv_path, markov_dir / "unknown_1.csv")
        _run([sys.executable, "read_float_notet.py", "unknown", "1"], cwd=markov_dir)
        unknown_m = markov_dir / "unknown_matrix.csv"
        shutil.move(str(unknown_m), str(matrices_dir / "unknown_matrix.csv"))
        _run([sys.executable, "read_matrix.py"], cwd=matrices_dir)

        line = results_log.read_text(encoding="utf-8").strip().splitlines()[-1]
        cols = [c for c in line.split("\t") if c.strip() != ""]
        # Column index 2 is Bhattacharyya, used by their statistics.py confusion matrix.
        pred_idx = int(cols[2])
        true_idx = EXTERNAL_ORDER.index(ref.ext_label)
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        rows.append(
            {
                "file": ref.source_path.name,
                "true_label": ref.ext_label,
                "pred_label": EXTERNAL_ORDER[pred_idx],
                "pred_idx": str(pred_idx),
            }
        )

        with expected_log.open("a", encoding="utf-8") as f:
            f.write(f"{true_idx}\n")

        _safe_unlink(matrices_dir / "unknown_matrix.csv")
        _safe_unlink(markov_dir / "unknown_1.csv")

    return y_true, y_pred, rows


def _metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark luciamarock Dastgah-Recognition-System on current dataset/splits.")
    p.add_argument("--external_repo", default=str(PROJECT_ROOT / "external" / "Dastgah-Recognition-System"))
    p.add_argument("--data", default=str(PROJECT_ROOT / "Training_Data"))
    p.add_argument("--splits", default=str(ROOT / "data" / "splits.json"))
    p.add_argument("--run_dir", default=str(ROOT / "runs" / "exp_luciamarock_benchmark_v1"))
    p.add_argument("--num_workers", type=int, default=4, help="parallel pitch extraction processes")
    p.add_argument("--resume", action="store_true", help="reuse extracted csvs from a previous run in the same run_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    external_repo = Path(args.external_repo).resolve()
    data_root = Path(args.data).resolve()
    splits_path = Path(args.splits).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    _check_essentia()
    if not external_repo.exists():
        raise FileNotFoundError(f"external repo not found: {external_repo}")
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits file not found: {splits_path}")

    work_repo = _prepare_worktree(external_repo, run_dir, resume=args.resume)
    train, test = _load_split_tracks(data_root, splits_path)
    train_refs, test_refs, n_failed = _stage_samples(work_repo, train, test, args.num_workers)
    _build_class_matrices(work_repo, train_refs)
    y_true, y_pred, pred_rows = _predict_test(work_repo, test_refs)

    m = _metrics(y_true, y_pred)
    print(f"External Markov Test: acc={m['acc']:.3f} macro_f1={m['macro_f1']:.3f} bal_acc={m['balanced_acc']:.3f}")

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test": m,
                "method": "luciamarock_markov_bhattacharyya",
                "external_repo": str(external_repo),
                "data": str(data_root),
                "splits": str(splits_path),
                "n_train_used": sum(len(v) for v in train_refs.values()),
                "n_test_classified": len(y_true),
                "n_extraction_failed": n_failed,
                "notes": "Uses external 7-class Markov pipeline with Bhattacharyya decision column.",
            },
            f,
            indent=2,
        )

    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "true_label", "pred_label", "pred_idx"])
        writer.writeheader()
        writer.writerows(pred_rows)

    print(f"Saved: {run_dir / 'metrics.json'}")
    print(f"Saved: {run_dir / 'predictions.csv'}")
    print(f"Work repo: {work_repo}")


if __name__ == "__main__":
    main()
