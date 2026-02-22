import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import librosa
import soundfile as sf
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dastgah_v2 import LABELS  # noqa: E402
from dastgah_v2.data import AUDIO_EXTENSIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True, help="Input dataset root (class folders inside)")
    p.add_argument("--output_root", required=True, help="Output dataset root for .wav files")
    p.add_argument("--sample_rate", type=int, default=22050, help="Target WAV sample rate")
    p.add_argument("--mono", action="store_true", help="Convert to mono (recommended for this project)")
    p.add_argument("--overwrite", action="store_true", help="Rewrite existing WAV files")
    p.add_argument("--num_workers", type=int, default=1, help="Conversion workers (processes)")
    return p.parse_args()


def _out_name(src_name: str, used: Dict[str, int]) -> str:
    base, ext = os.path.splitext(src_name)
    key = base.lower()
    if key not in used:
        used[key] = 0
        return f"{base}.wav"
    used[key] += 1
    digest = hashlib.md5(src_name.encode("utf-8")).hexdigest()[:6]
    return f"{base}__{ext.lower().lstrip('.')}__{digest}.wav"


def collect_jobs(input_root: str, output_root: str) -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []
    for label in LABELS:
        in_dir = os.path.join(input_root, label)
        if not os.path.isdir(in_dir):
            raise FileNotFoundError(f"Missing class folder: {in_dir}")
        out_dir = os.path.join(output_root, label)
        os.makedirs(out_dir, exist_ok=True)

        used_names: Dict[str, int] = {}
        for name in sorted(os.listdir(in_dir)):
            if not name.lower().endswith(AUDIO_EXTENSIONS):
                continue
            src = os.path.join(in_dir, name)
            dst = os.path.join(out_dir, _out_name(name, used_names))
            jobs.append((src, dst))
    return jobs


def _convert_one(job: Tuple[str, str, int, bool, bool]) -> Tuple[str, str]:
    src, dst, sr, mono, overwrite = job
    if (not overwrite) and os.path.exists(dst):
        src_sig = os.stat(src)
        dst_sig = os.stat(dst)
        if dst_sig.st_mtime_ns >= src_sig.st_mtime_ns and dst_sig.st_size > 0:
            return "skipped", dst

    audio, _ = librosa.load(src, sr=sr, mono=mono)
    if audio.ndim == 1:
        sf.write(dst, audio, sr, subtype="PCM_16")
    else:
        sf.write(dst, audio.T, sr, subtype="PCM_16")
    return "converted", dst


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    jobs = collect_jobs(args.input_root, args.output_root)
    if not jobs:
        print("No audio files found to convert.")
        return

    converted = 0
    skipped = 0

    worker_jobs = [(src, dst, args.sample_rate, args.mono, args.overwrite) for src, dst in jobs]
    if args.num_workers <= 1:
        iterator = tqdm(worker_jobs, desc="Converting audio")
        for job in iterator:
            status, _ = _convert_one(job)
            if status == "converted":
                converted += 1
            else:
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(_convert_one, job): job for job in worker_jobs}
            iterator = tqdm(as_completed(futs), total=len(futs), desc="Converting audio")
            for fut in iterator:
                src = futs[fut][0]
                try:
                    status, _ = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed converting file: {src}") from exc
                if status == "converted":
                    converted += 1
                else:
                    skipped += 1

    print(f"Done. converted={converted} skipped={skipped} total={len(worker_jobs)}")


if __name__ == "__main__":
    main()
