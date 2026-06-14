from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from soulx_duplug.data.audio import audio_metadata
from soulx_duplug.data.manifest import Stage1Record, assign_dev_split, coarse_split, group_by_split, write_manifest
from soulx_duplug.data.profiles import enabled_dataset_ids, load_dataset_profile
from soulx_duplug.data.text import normalize_text


DEFAULT_DATA_ROOT = Path("/root/SoulX-Duplug/datasets")
LOCAL_DEFAULT_DATASETS = ("aishell1", "aishell3")
SUPPORTED_DATASETS = (
    "aishell1",
    "aishell3",
    "wenetspeech",
    "commonvoice-cn",
    "emilia-cn",
    "magicdata",
    "librispeech",
    "gigaspeech",
    "commonvoice-en",
    "emilia-en",
)
PAPER_ASR_DATASETS = SUPPORTED_DATASETS


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg")


@dataclass(frozen=True)
class SelectionSummary:
    target_hours: dict[str, float]
    available_hours: dict[str, float]
    selected_hours: dict[str, float]
    selected_records: dict[str, int]
    skipped_missing_duration: dict[str, int]
    selected_by_dataset: dict[str, dict[str, Any]]
    warnings: list[str]


def _safe_extract_tar(path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_dir)):
                raise RuntimeError(f"unsafe tar member in {path}: {member.name}")
        tar.extractall(target_dir)


def maybe_extract_aishell1_nested_archives(dataset_root: Path, *, delete_archives: bool = False) -> None:
    wav_dir = dataset_root / "extracted" / "data_aishell" / "wav"
    if not wav_dir.exists():
        return
    for archive in sorted(wav_dir.glob("*.tar.gz")):
        marker = wav_dir / f".{archive.name}.done"
        if marker.exists():
            continue
        print(f"[extract] {archive}")
        _safe_extract_tar(archive, wav_dir)
        marker.write_text("done\n", encoding="utf-8")
        if delete_archives:
            archive.unlink()


def prepare_nested_archives(data_root: Path, datasets: list[str], *, delete_archives: bool = False) -> None:
    if "aishell1" in datasets:
        maybe_extract_aishell1_nested_archives(data_root / "aishell1", delete_archives=delete_archives)


def index_audio_files(base_dir: Path, extensions: tuple[str, ...] = (".wav", ".flac")) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not base_dir.exists():
        return index
    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index.setdefault(path.stem, path)
    return index


def candidate_roots(data_root: Path, dataset: str, *extra: str) -> list[Path]:
    names = [dataset, dataset.replace("-", "_"), *extra]
    roots: list[Path] = []
    for name in names:
        path = data_root / name
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def resolve_audio_path(relative_or_absolute: str | None, bases: Iterable[Path]) -> Path | None:
    if not relative_or_absolute:
        return None
    raw = Path(str(relative_or_absolute))
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = []
    for base in bases:
        candidates.extend(
            [
                base / raw,
                base / "clips" / raw,
                base / "wav" / raw,
                base / "wavs" / raw,
                base / "audio" / raw,
                base / "mp3" / raw,
                base / "extracted" / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def open_text_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def _text_from_obj(obj: dict[str, Any]) -> str:
    for key in ("text", "txt", "sentence", "transcript", "normalized_text", "raw_text"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def _audio_from_obj(obj: dict[str, Any]) -> str | None:
    for key in ("wav", "audio", "audio_path", "path", "file", "filepath", "clip"):
        value = obj.get(key)
        if value:
            return str(value)
    return None


def _speaker_from_obj(obj: dict[str, Any]) -> str:
    for key in ("speaker", "speaker_id", "client_id", "spk", "spk_id"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def _duration_from_obj(obj: dict[str, Any]) -> float | None:
    for key in ("duration", "duration_sec", "length", "audio_duration"):
        value = obj.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _time_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result > 1000.0:
        result /= 1000.0
    return result


def _segment_bounds(obj: dict[str, Any]) -> tuple[float | None, float | None]:
    start = None
    end = None
    for key in ("start", "begin", "begin_time", "start_time", "start_sec"):
        start = _time_to_seconds(obj.get(key))
        if start is not None:
            break
    for key in ("end", "end_time", "stop", "stop_time", "end_sec"):
        end = _time_to_seconds(obj.get(key))
        if end is not None:
            break
    return start, end


def _split_from_subset(value: Any) -> str:
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    elif value is None:
        text = ""
    else:
        text = str(value)
    lowered = text.lower()
    if "dev" in lowered or "valid" in lowered:
        return "dev"
    if "test" in lowered or "eval" in lowered:
        return "test"
    return coarse_split(lowered or "train")


def _record_id(dataset: str, obj: dict[str, Any], audio_path: Path, index: int) -> str:
    for key in ("utt_id", "id", "sid", "segment_id", "key", "aid"):
        value = obj.get(key)
        if value:
            return str(value)
    return f"{dataset}-{audio_path.stem}-{index:08d}"


def _record_with_optional_metadata(record: Stage1Record, read_metadata: bool) -> Stage1Record:
    if not read_metadata:
        return record
    try:
        metadata = audio_metadata(record.audio_path)
        duration = metadata.duration
        if record.segment_start is not None and record.segment_end is not None:
            duration = max(0.0, record.segment_end - record.segment_start)
        return record.with_audio_info(
            duration=duration,
            sample_rate=metadata.sample_rate,
            channels=metadata.channels,
        )
    except Exception:
        return record


def _split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for split in ("train", "dev", "test", "valid", "validation"):
        if split in parts:
            return coarse_split(split)
    name = path.name.lower()
    if name.startswith(("dev", "test", "train")):
        return coarse_split(name)
    return "train"


def parse_aishell1(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    dataset_root = data_root / "aishell1"
    base = dataset_root / "extracted" / "data_aishell"
    transcript = base / "transcript" / "aishell_transcript_v0.8.txt"
    if not transcript.exists():
        print(f"[skip] AISHELL-1 transcript not found: {transcript}")
        return []
    audio_index = index_audio_files(base / "wav", extensions=(".wav",))
    if not audio_index and list((base / "wav").glob("*.tar.gz")):
        print("[warn] AISHELL-1 inner speaker archives are not extracted. Re-run with --extract-nested-archives.")

    records: list[Stage1Record] = []
    missing = 0
    with transcript.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            audio_path = audio_index.get(utt_id)
            text = normalize_text(text, "zh", dataset="aishell1")
            if not audio_path or not text:
                missing += 1
                continue
            speaker_id = utt_id.split("W", 1)[0] if "W" in utt_id else ""
            record = Stage1Record(
                utt_id=utt_id,
                dataset="aishell1",
                split=_split_from_path(audio_path),
                lang="zh",
                audio_path=str(audio_path),
                text=text,
                speaker_id=speaker_id,
            )
            records.append(_record_with_optional_metadata(record, read_metadata))
    if missing:
        print(f"[warn] AISHELL-1 skipped {missing} transcript rows without audio/text")
    return records


def _parse_aishell3_content(content_path: Path, audio_index: dict[str, Path], split: str, read_metadata: bool) -> list[Stage1Record]:
    records: list[Stage1Record] = []
    missing = 0
    with content_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            filename, text = parts
            utt_id = Path(filename).stem
            audio_path = audio_index.get(utt_id)
            text = normalize_text(text, "zh", dataset="aishell3")
            if not audio_path or not text:
                missing += 1
                continue
            speaker_id = audio_path.parent.name
            record = Stage1Record(
                utt_id=utt_id,
                dataset="aishell3",
                split=coarse_split(split),
                lang="zh",
                audio_path=str(audio_path),
                text=text,
                speaker_id=speaker_id,
            )
            records.append(_record_with_optional_metadata(record, read_metadata))
    if missing:
        print(f"[warn] AISHELL-3 {split} skipped {missing} rows without audio/text")
    return records


def parse_aishell3(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    base = data_root / "aishell3" / "extracted"
    if not base.exists():
        print(f"[skip] AISHELL-3 extracted dir not found: {base}")
        return []
    records: list[Stage1Record] = []
    for split_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        content = split_dir / "content.txt"
        wav_dir = split_dir / "wav"
        if not content.exists() or not wav_dir.exists():
            continue
        audio_index = index_audio_files(wav_dir, extensions=(".wav",))
        records.extend(_parse_aishell3_content(content, audio_index, split_dir.name, read_metadata))
    return records


def parse_librispeech(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    dataset_root = data_root / "librispeech"
    candidates = [
        dataset_root / "extracted" / "LibriSpeech",
        dataset_root / "LibriSpeech",
        dataset_root / "extracted",
        dataset_root,
    ]
    base = next((path for path in candidates if path.exists() and list(path.rglob("*.trans.txt"))), None)
    if base is None:
        print(f"[skip] LibriSpeech extracted transcripts not found under {dataset_root}")
        return []

    records: list[Stage1Record] = []
    for transcript in sorted(base.rglob("*.trans.txt")):
        audio_index = index_audio_files(transcript.parent, extensions=(".flac", ".wav"))
        split = _split_from_path(transcript)
        speaker_id = transcript.parent.parent.name if transcript.parent.parent != base else ""
        with transcript.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                audio_path = audio_index.get(utt_id)
                text = normalize_text(text, "en", dataset="librispeech")
                if not audio_path or not text:
                    continue
                record = Stage1Record(
                    utt_id=utt_id,
                    dataset="librispeech",
                    split=split,
                    lang="en",
                    audio_path=str(audio_path),
                    text=text,
                    speaker_id=speaker_id,
                )
                records.append(_record_with_optional_metadata(record, read_metadata))
    return records


def _iter_magicdata_transcripts(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.rglob("*.txt")):
        lowered = path.name.lower()
        if lowered in {".download-state.json"} or "readme" in lowered:
            continue
        if lowered == "trans.txt" or "trans" in lowered or "text" in lowered:
            yield path


def parse_magicdata(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    dataset_root = data_root / "magicdata"
    if not dataset_root.exists():
        print(f"[skip] MAGICDATA dir not found: {dataset_root}")
        return []
    audio_index = index_audio_files(dataset_root, extensions=(".wav",))
    if not audio_index:
        print(f"[skip] MAGICDATA extracted wav files not found under {dataset_root}")
        return []

    records: list[Stage1Record] = []
    seen: set[str] = set()
    for transcript in _iter_magicdata_transcripts(dataset_root):
        split = _split_from_path(transcript)
        with transcript.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_or_file, text = parts
                utt_id = Path(utt_or_file).stem
                audio_path = audio_index.get(utt_id)
                text = normalize_text(text, "zh", dataset="magicdata")
                if not audio_path or not text or utt_id in seen:
                    continue
                seen.add(utt_id)
                speaker_id = audio_path.parent.name
                record = Stage1Record(
                    utt_id=utt_id,
                    dataset="magicdata",
                    split=split,
                    lang="zh",
                    audio_path=str(audio_path),
                    text=text,
                    speaker_id=speaker_id,
                )
                records.append(_record_with_optional_metadata(record, read_metadata))
    return records


def parse_commonvoice_dataset(
    data_root: Path,
    *,
    dataset: str,
    lang: str,
    read_metadata: bool = False,
) -> list[Stage1Record]:
    roots = candidate_roots(data_root, dataset, "commonvoice", "common_voice", "voxbox")
    if not roots:
        print(f"[skip] {dataset} dir not found under {data_root}")
        return []

    records: list[Stage1Record] = []
    seen: set[str] = set()
    for root in roots:
        for tsv_path in sorted(root.rglob("*.tsv")):
            split = _split_from_path(tsv_path)
            try:
                with tsv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    if not reader.fieldnames or "sentence" not in reader.fieldnames:
                        continue
                    for idx, row in enumerate(reader, start=1):
                        rel_audio = row.get("path") or row.get("audio") or row.get("clip")
                        audio_path = resolve_audio_path(rel_audio, [tsv_path.parent, root])
                        text = normalize_text(row.get("sentence", ""), lang, dataset=dataset)
                        if not audio_path or not text:
                            continue
                        utt_id = str(row.get("client_id") or Path(str(rel_audio)).stem or f"{tsv_path.stem}-{idx}")
                        utt_id = f"{utt_id}-{Path(str(rel_audio)).stem}" if row.get("client_id") else utt_id
                        key = f"{dataset}:{utt_id}:{audio_path}"
                        if key in seen:
                            continue
                        seen.add(key)
                        record = Stage1Record(
                            utt_id=utt_id,
                            dataset=dataset,
                            split=split,
                            lang=lang,
                            audio_path=str(audio_path),
                            text=text,
                            speaker_id=str(row.get("client_id") or ""),
                            duration=_duration_from_obj(row),
                        )
                        records.append(_record_with_optional_metadata(record, read_metadata))
            except OSError as exc:
                print(f"[warn] failed to read Common Voice TSV {tsv_path}: {exc}")
    print(f"[manifest] {dataset}: parsed {len(records)} Common Voice records")
    return records


def parse_commonvoice_cn(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    return parse_commonvoice_dataset(data_root, dataset="commonvoice-cn", lang="zh", read_metadata=read_metadata)


def parse_commonvoice_en(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    return parse_commonvoice_dataset(data_root, dataset="commonvoice-en", lang="en", read_metadata=read_metadata)


def parse_emilia_dataset(
    data_root: Path,
    *,
    dataset: str,
    lang: str,
    read_metadata: bool = False,
) -> list[Stage1Record]:
    lang_tags = {"zh": {"zh", "cn", "chinese", "mandarin"}, "en": {"en", "eng", "english"}}[lang]
    roots = candidate_roots(data_root, dataset, "emilia", "Emilia", "voxbox")
    if not roots:
        print(f"[skip] {dataset} dir not found under {data_root}")
        return []

    records: list[Stage1Record] = []
    seen: set[str] = set()
    for root in roots:
        metadata_paths = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.jsonl.gz"))
        for metadata_path in metadata_paths:
            for idx, obj in enumerate(iter_jsonl(metadata_path), start=1):
                obj_lang = str(obj.get("language") or obj.get("lang") or "").lower()
                if obj_lang and obj_lang not in lang_tags:
                    continue
                rel_audio = _audio_from_obj(obj)
                audio_path = resolve_audio_path(rel_audio, [metadata_path.parent, root])
                text = normalize_text(_text_from_obj(obj), lang, dataset=dataset)
                if not audio_path or not text:
                    continue
                utt_id = _record_id(dataset, obj, audio_path, idx)
                key = f"{dataset}:{utt_id}:{audio_path}"
                if key in seen:
                    continue
                seen.add(key)
                record = Stage1Record(
                    utt_id=utt_id,
                    dataset=dataset,
                    split=_split_from_path(metadata_path),
                    lang=lang,
                    audio_path=str(audio_path),
                    text=text,
                    speaker_id=_speaker_from_obj(obj),
                    duration=_duration_from_obj(obj),
                )
                records.append(_record_with_optional_metadata(record, read_metadata))

        for json_path in sorted(root.rglob("*.json")):
            if json_path.name.lower() in {"configuration.json", "dataset_infos.json"}:
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            obj_lang = str(data.get("language") or data.get("lang") or "").lower()
            if obj_lang and obj_lang not in lang_tags:
                continue
            text = normalize_text(_text_from_obj(data), lang, dataset=dataset)
            if not text:
                continue
            rel_audio = _audio_from_obj(data)
            audio_path = resolve_audio_path(rel_audio, [json_path.parent, root])
            if audio_path is None:
                for suffix in AUDIO_EXTENSIONS:
                    candidate = json_path.with_suffix(suffix)
                    if candidate.exists():
                        audio_path = candidate
                        break
            if audio_path is None:
                continue
            utt_id = _record_id(dataset, data, audio_path, 0)
            key = f"{dataset}:{utt_id}:{audio_path}"
            if key in seen:
                continue
            seen.add(key)
            record = Stage1Record(
                utt_id=utt_id,
                dataset=dataset,
                split=_split_from_path(json_path),
                lang=lang,
                audio_path=str(audio_path),
                text=text,
                speaker_id=_speaker_from_obj(data),
                duration=_duration_from_obj(data),
            )
            records.append(_record_with_optional_metadata(record, read_metadata))
    print(f"[manifest] {dataset}: parsed {len(records)} Emilia records")
    return records


def parse_emilia_cn(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    return parse_emilia_dataset(data_root, dataset="emilia-cn", lang="zh", read_metadata=read_metadata)


def parse_emilia_en(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    return parse_emilia_dataset(data_root, dataset="emilia-en", lang="en", read_metadata=read_metadata)


def _iter_json_metadata(root: Path, preferred_names: tuple[str, ...]) -> Iterator[Path]:
    preferred = {name.lower() for name in preferred_names}
    for path in sorted(root.rglob("*.json")):
        lowered = path.name.lower()
        if preferred and lowered not in preferred and not any(name in lowered for name in preferred):
            continue
        yield path


def _segments_from_audio_obj(audio_obj: dict[str, Any]) -> list[dict[str, Any]]:
    segments = audio_obj.get("segments")
    if isinstance(segments, list):
        return [segment for segment in segments if isinstance(segment, dict)]
    if isinstance(audio_obj.get("segment"), dict):
        return [audio_obj["segment"]]
    return []


def _audio_entries_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("audios", "audio", "data", "items", "utterances"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if _segments_from_audio_obj(data) or _audio_from_obj(data):
            return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def parse_segmented_json_dataset(
    data_root: Path,
    *,
    dataset: str,
    lang: str,
    root_names: tuple[str, ...],
    preferred_metadata_names: tuple[str, ...],
    read_metadata: bool = False,
) -> list[Stage1Record]:
    roots: list[Path] = []
    for root_name in root_names:
        roots.extend(candidate_roots(data_root, root_name))
    if not roots:
        print(f"[skip] {dataset} dir not found under {data_root}")
        return []

    records: list[Stage1Record] = []
    seen: set[str] = set()
    for root in roots:
        for metadata_path in _iter_json_metadata(root, preferred_metadata_names):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[warn] failed to read {dataset} metadata {metadata_path}: {exc}")
                continue
            for audio_idx, audio_obj in enumerate(_audio_entries_from_json(data), start=1):
                audio_rel = _audio_from_obj(audio_obj)
                audio_path = resolve_audio_path(audio_rel, [metadata_path.parent, root])
                if audio_path is None:
                    continue
                audio_split = _split_from_subset(
                    audio_obj.get("subsets")
                    or audio_obj.get("subset")
                    or audio_obj.get("split")
                    or metadata_path
                )
                audio_speaker = _speaker_from_obj(audio_obj)
                segments = _segments_from_audio_obj(audio_obj)
                if not segments:
                    segments = [audio_obj]
                for seg_idx, segment in enumerate(segments, start=1):
                    text = normalize_text(_text_from_obj(segment), lang, dataset=dataset)
                    if not text:
                        continue
                    start, end = _segment_bounds(segment)
                    duration = _duration_from_obj(segment)
                    if duration is None and start is not None and end is not None:
                        duration = max(0.0, end - start)
                    split = _split_from_subset(
                        segment.get("subsets")
                        or segment.get("subset")
                        or segment.get("split")
                        or audio_split
                    )
                    utt_id = _record_id(dataset, segment, audio_path, audio_idx * 1_000_000 + seg_idx)
                    key = f"{dataset}:{utt_id}:{audio_path}:{start}:{end}"
                    if key in seen:
                        continue
                    seen.add(key)
                    record = Stage1Record(
                        utt_id=utt_id,
                        dataset=dataset,
                        split=split,
                        lang=lang,
                        audio_path=str(audio_path),
                        text=text,
                        speaker_id=_speaker_from_obj(segment) or audio_speaker,
                        duration=duration,
                        segment_start=start,
                        segment_end=end,
                    )
                    records.append(_record_with_optional_metadata(record, read_metadata))
    print(f"[manifest] {dataset}: parsed {len(records)} segmented records")
    return records


def parse_wenetspeech(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    return parse_segmented_json_dataset(
        data_root,
        dataset="wenetspeech",
        lang="zh",
        root_names=("wenetspeech", "WenetSpeech"),
        preferred_metadata_names=("wenetspeech.json", "metadata.json", "data.json"),
        read_metadata=read_metadata,
    )


def parse_gigaspeech(data_root: Path, *, read_metadata: bool = False) -> list[Stage1Record]:
    return parse_segmented_json_dataset(
        data_root,
        dataset="gigaspeech",
        lang="en",
        root_names=("gigaspeech", "GigaSpeech"),
        preferred_metadata_names=("gigaspeech.json", "metadata.json"),
        read_metadata=read_metadata,
    )


PARSERS = {
    "aishell1": parse_aishell1,
    "aishell3": parse_aishell3,
    "wenetspeech": parse_wenetspeech,
    "commonvoice-cn": parse_commonvoice_cn,
    "emilia-cn": parse_emilia_cn,
    "librispeech": parse_librispeech,
    "magicdata": parse_magicdata,
    "gigaspeech": parse_gigaspeech,
    "commonvoice-en": parse_commonvoice_en,
    "emilia-en": parse_emilia_en,
}


def _selection_key(record: Stage1Record, seed: str) -> str:
    raw = "|".join(
        [
            seed,
            record.lang,
            record.dataset,
            record.utt_id,
            record.audio_path,
            "" if record.segment_start is None else f"{record.segment_start:.3f}",
            "" if record.segment_end is None else f"{record.segment_end:.3f}",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _duration_seconds(record: Stage1Record) -> float | None:
    if record.duration is not None:
        try:
            duration = float(record.duration)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0.0:
            return duration
    if record.segment_start is not None and record.segment_end is not None:
        return max(0.0, float(record.segment_end) - float(record.segment_start))
    return None


def _hours(seconds: float) -> float:
    return round(seconds / 3600.0, 4)


def select_records_by_target_hours(
    records: list[Stage1Record],
    *,
    target_hours: dict[str, float],
    seed: str = "paper-target-v1",
    split: str | None = "train",
) -> tuple[list[Stage1Record], SelectionSummary]:
    target_seconds = {lang: float(hours) * 3600.0 for lang, hours in target_hours.items() if float(hours) > 0.0}
    if not target_seconds:
        return records, SelectionSummary(
            target_hours={},
            available_hours={},
            selected_hours={},
            selected_records={},
            skipped_missing_duration={},
            selected_by_dataset={},
            warnings=[],
        )

    passthrough: list[Stage1Record] = []
    eligible_by_lang: dict[str, list[tuple[Stage1Record, float]]] = {lang: [] for lang in target_seconds}
    skipped_missing_duration: Counter[str] = Counter()
    warnings: list[str] = []

    for record in records:
        if split is not None and record.split != split:
            passthrough.append(record)
            continue
        if record.lang not in target_seconds:
            passthrough.append(record)
            continue
        duration = _duration_seconds(record)
        if duration is None or duration <= 0.0:
            skipped_missing_duration[record.lang] += 1
            continue
        eligible_by_lang[record.lang].append((record, duration))

    selected: list[Stage1Record] = list(passthrough)
    available_seconds: dict[str, float] = {}
    selected_seconds: dict[str, float] = {}
    selected_records: dict[str, int] = {}
    selected_by_dataset: dict[str, dict[str, float | int]] = {}

    for lang, target in target_seconds.items():
        eligible = eligible_by_lang.get(lang, [])
        available = sum(duration for _, duration in eligible)
        available_seconds[lang] = available
        if not eligible:
            warnings.append(f"{lang}: no records with positive duration; selected 0h against target {_hours(target)}h")
            selected_seconds[lang] = 0.0
            selected_records[lang] = 0
            continue
        if available <= target:
            lang_selected = [record for record, _ in eligible]
            if available < target:
                warnings.append(f"{lang}: available {_hours(available)}h is below target {_hours(target)}h; kept all available records")
        else:
            by_dataset: dict[str, list[tuple[Stage1Record, float]]] = {}
            for record, duration in eligible:
                by_dataset.setdefault(record.dataset, []).append((record, duration))

            selected_keys: set[str] = set()
            lang_selected_pairs: list[tuple[Stage1Record, float]] = []
            for dataset, dataset_items in sorted(by_dataset.items()):
                dataset_total = sum(duration for _, duration in dataset_items)
                dataset_quota = target * dataset_total / available
                running = 0.0
                for record, duration in sorted(dataset_items, key=lambda item: _selection_key(item[0], seed)):
                    if running + duration > dataset_quota:
                        continue
                    key = _selection_key(record, seed)
                    selected_keys.add(key)
                    lang_selected_pairs.append((record, duration))
                    running += duration

            selected_total = sum(duration for _, duration in lang_selected_pairs)
            for record, duration in sorted(eligible, key=lambda item: _selection_key(item[0], f"{seed}:fill")):
                if selected_total >= target:
                    break
                key = _selection_key(record, seed)
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                lang_selected_pairs.append((record, duration))
                selected_total += duration
            lang_selected = [record for record, _ in lang_selected_pairs]

        selected.extend(lang_selected)
        selected_duration = sum(_duration_seconds(record) or 0.0 for record in lang_selected)
        selected_seconds[lang] = selected_duration
        selected_records[lang] = len(lang_selected)
        for record in lang_selected:
            duration = _duration_seconds(record) or 0.0
            item = selected_by_dataset.setdefault(record.dataset, {"lang": record.lang, "records": 0, "hours": 0.0})
            item["records"] = int(item["records"]) + 1
            item["hours"] = float(item["hours"]) + duration / 3600.0

    for item in selected_by_dataset.values():
        item["hours"] = round(float(item["hours"]), 4)

    summary = SelectionSummary(
        target_hours={lang: _hours(seconds) for lang, seconds in target_seconds.items()},
        available_hours={lang: _hours(seconds) for lang, seconds in available_seconds.items()},
        selected_hours={lang: _hours(seconds) for lang, seconds in selected_seconds.items()},
        selected_records=selected_records,
        skipped_missing_duration=dict(sorted(skipped_missing_duration.items())),
        selected_by_dataset=dict(sorted(selected_by_dataset.items())),
        warnings=warnings,
    )
    return selected, summary


def write_selection_summary(summary: SelectionSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_manifest(
    *,
    data_root: Path,
    datasets: list[str],
    read_metadata: bool = False,
    dev_ratio: float = 0.01,
    max_samples_per_dataset: int | None = None,
    dedup: str = "source_text",
    target_hours: dict[str, float] | None = None,
    selection_seed: str = "paper-target-v1",
    selection_summary_path: str | Path | None = None,
) -> list[Stage1Record]:
    records: list[Stage1Record] = []
    for dataset in datasets:
        if dataset not in PARSERS:
            raise ValueError(f"unsupported dataset {dataset}; choose from {', '.join(SUPPORTED_DATASETS)}")
        dataset_records = PARSERS[dataset](data_root, read_metadata=read_metadata)
        dataset_records.sort(key=lambda item: item.utt_id)
        if max_samples_per_dataset is not None:
            dataset_records = dataset_records[:max_samples_per_dataset]
        print(f"[manifest] {dataset}: {len(dataset_records)} records")
        records.extend(dataset_records)
    records = deduplicate_records(records, mode=dedup)
    records = assign_dev_split(records, dev_ratio)
    if target_hours:
        records, summary = select_records_by_target_hours(records, target_hours=target_hours, seed=selection_seed)
        print(f"[selection] target_hours={summary.target_hours}")
        print(f"[selection] selected_hours={summary.selected_hours}; selected_records={summary.selected_records}")
        if summary.skipped_missing_duration:
            print(f"[selection] skipped_missing_duration={summary.skipped_missing_duration}")
        for warning in summary.warnings:
            print(f"[warn] {warning}")
        if selection_summary_path:
            write_selection_summary(summary, selection_summary_path)
            print(f"[write] {selection_summary_path}")
    return records


def deduplicate_records(records: list[Stage1Record], *, mode: str) -> list[Stage1Record]:
    if mode in {"none", "off", ""}:
        return records
    seen: set[str] = set()
    result: list[Stage1Record] = []
    skipped = 0
    for record in records:
        if mode == "utt":
            key = f"{record.dataset}:{record.utt_id}"
        elif mode == "source_text":
            audio_id = Path(record.audio_path).stem
            start = "" if record.segment_start is None else f"{record.segment_start:.3f}"
            end = "" if record.segment_end is None else f"{record.segment_end:.3f}"
            key = f"{audio_id}:{start}:{end}:{record.text}"
        else:
            raise ValueError("dedup must be one of: none, utt, source_text")
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        result.append(record)
    if skipped:
        print(f"[dedup] skipped {skipped} duplicate records by mode={mode}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 1 ASR JSONL manifests.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", default=",".join(LOCAL_DEFAULT_DATASETS))
    parser.add_argument("--profile", type=Path, help="Dataset profile YAML. Overrides --datasets when provided.")
    parser.add_argument("--out-dir", type=Path, default=Path("manifests/stage1"))
    parser.add_argument("--dev-ratio", type=float, default=0.01)
    parser.add_argument("--max-samples-per-dataset", type=int)
    parser.add_argument("--with-audio-metadata", action="store_true")
    parser.add_argument("--dedup", choices=("none", "utt", "source_text"), default=None)
    parser.add_argument("--selection-seed", default="paper-target-v1")
    parser.add_argument("--selection-summary", type=Path, help="Write paper target-hour selection summary JSON.")
    parser.add_argument("--disable-target-hours", action="store_true", help="Ignore target_hours declared in the profile.")
    parser.add_argument("--prepare-archives", action="store_true", help="Extract dataset-internal archives before manifest creation.")
    parser.add_argument("--extract-nested-archives", action="store_true", help="Backward-compatible alias for --prepare-archives.")
    parser.add_argument("--delete-nested-archives", action="store_true", help="Delete nested archives after successful extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dedup = args.dedup or "source_text"
    target_hours: dict[str, float] = {}
    if args.profile:
        profile = load_dataset_profile(args.profile)
        datasets = enabled_dataset_ids(profile)
        dedup = args.dedup or profile.dedup
        print(f"[profile] {profile.name}: {', '.join(datasets)}")
        if not args.disable_target_hours:
            target_hours = dict(profile.target_hours)
    else:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if target_hours and not args.with_audio_metadata:
        raise RuntimeError("profile target_hours requires --with-audio-metadata so the selector can match paper hours")
    if args.prepare_archives or args.extract_nested_archives:
        prepare_nested_archives(args.data_root, datasets, delete_archives=args.delete_nested_archives)
    selection_summary_path = args.selection_summary
    if target_hours and selection_summary_path is None:
        selection_summary_path = args.out_dir / "selection.summary.json"
    records = build_manifest(
        data_root=args.data_root,
        datasets=datasets,
        read_metadata=args.with_audio_metadata,
        dev_ratio=args.dev_ratio,
        max_samples_per_dataset=args.max_samples_per_dataset,
        dedup=dedup,
        target_hours=target_hours,
        selection_seed=args.selection_seed,
        selection_summary_path=selection_summary_path,
    )
    grouped = group_by_split(records)
    for split, split_records in grouped.items():
        path = args.out_dir / f"{split}.jsonl"
        count = write_manifest(split_records, path)
        print(f"[write] {path}: {count}")
    print(f"[summary] {dict(Counter(record.split for record in records))}")


if __name__ == "__main__":
    main()
