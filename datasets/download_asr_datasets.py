#!/usr/bin/env python3
"""Download ASR datasets with resume support.

The script intentionally stays self-contained: no aria2, git-lfs, or
huggingface-cli is required.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import errno
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = Path(__file__).resolve().parent
NETWORK_TURBO = Path("/etc/network_turbo")
STATE_FILE = ".download-state.json"
USER_AGENT = "soulx-duplug-dataset-downloader/1.0"
HF_ENDPOINT = "https://huggingface.co"
WENETSPEECH_GIT_URL = "https://github.com/wenet-e2e/WenetSpeech.git"
LOG_FILE: Path | None = None


@dataclass(frozen=True)
class FileSpec:
    key: str
    filename: str
    description: str
    urls: tuple[str, ...]
    size_bytes: int | None = None
    display_size: str = "unknown"
    md5: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    source: str
    kind: str
    files: tuple[FileSpec, ...] = ()
    repo_id: str | None = None
    revision: str = "main"
    hf_prefixes: tuple[str, ...] = ()
    hf_includes: tuple[str, ...] = ()
    hf_excludes: tuple[str, ...] = ()
    requires_filter: bool = False
    requires_auth: bool = False
    manual_message: str = ""


@dataclass(frozen=True)
class DownloadTask:
    logical_id: str
    source_dataset: str
    dedup_key: str
    files: str = "all"
    revision: str | None = None
    hf_prefix: tuple[str, ...] = ()
    hf_include: tuple[str, ...] = ()
    hf_exclude: tuple[str, ...] = ()
    hf_all: bool = False
    max_files: int | None = None


def openslr_file(resource_id: int, key: str, filename: str, desc: str, size: str) -> FileSpec:
    mirrors = (
        f"https://openslr.trmal.net/resources/{resource_id}/{filename}",
        f"https://openslr.elda.org/resources/{resource_id}/{filename}",
        f"https://www.openslr.org/resources/{resource_id}/{filename}",
    )
    return FileSpec(key=key, filename=filename, description=desc, urls=mirrors, display_size=size)


DATASETS: dict[str, DatasetSpec] = {
    "aishell1": DatasetSpec(
        key="aishell1",
        name="AISHELL-1",
        source="OpenSLR SLR33: https://www.openslr.org/33/",
        kind="http",
        files=(
            FileSpec(
                key="data",
                filename="data_aishell.tgz",
                description="speech data and transcripts",
                urls=(
                    "https://openslr.trmal.net/resources/33/data_aishell.tgz",
                    "https://openslr.elda.org/resources/33/data_aishell.tgz",
                    "https://www.openslr.org/resources/33/data_aishell.tgz",
                ),
                size_bytes=15_582_913_665,
                display_size="15G",
            ),
            FileSpec(
                key="resource",
                filename="resource_aishell.tgz",
                description="supplementary resources, lexicon, speaker info",
                urls=(
                    "https://openslr.trmal.net/resources/33/resource_aishell.tgz",
                    "https://openslr.elda.org/resources/33/resource_aishell.tgz",
                    "https://www.openslr.org/resources/33/resource_aishell.tgz",
                ),
                size_bytes=1_246_920,
                display_size="1.2M",
            ),
        ),
    ),
    "aishell3": DatasetSpec(
        key="aishell3",
        name="AISHELL-3",
        source="OpenSLR SLR93: https://www.openslr.org/93/",
        kind="http",
        files=(
            openslr_file(
                93,
                "data",
                "data_aishell3.tgz",
                "speech data and transcripts",
                "19G",
            ),
        ),
    ),
    "magicdata": DatasetSpec(
        key="magicdata",
        name="MAGICDATA Mandarin Chinese Read Speech Corpus",
        source="OpenSLR SLR68: https://www.openslr.org/68/",
        kind="http",
        files=(
            openslr_file(68, "train", "train_set.tar.gz", "training set", "52G"),
            openslr_file(68, "dev", "dev_set.tar.gz", "development set", "1.0G"),
            openslr_file(68, "test", "test_set.tar.gz", "test set", "2.2G"),
            openslr_file(68, "metadata", "metadata.tar.gz", "metadata and speaker info", "3.8M"),
        ),
    ),
    "librispeech": DatasetSpec(
        key="librispeech",
        name="LibriSpeech ASR corpus",
        source="OpenSLR SLR12: https://www.openslr.org/12/",
        kind="http",
        files=(
            openslr_file(12, "dev-clean", "dev-clean.tar.gz", "development clean", "337M"),
            openslr_file(12, "dev-other", "dev-other.tar.gz", "development other", "314M"),
            openslr_file(12, "test-clean", "test-clean.tar.gz", "test clean", "346M"),
            openslr_file(12, "test-other", "test-other.tar.gz", "test other", "328M"),
            openslr_file(12, "train-clean-100", "train-clean-100.tar.gz", "train clean 100h", "6.3G"),
            openslr_file(12, "train-clean-360", "train-clean-360.tar.gz", "train clean 360h", "23G"),
            openslr_file(12, "train-other-500", "train-other-500.tar.gz", "train other 500h", "30G"),
            openslr_file(12, "md5sum", "md5sum.txt", "MD5 checksums", "600B"),
        ),
    ),
    "emilia-cn": DatasetSpec(
        key="emilia-cn",
        name="Emilia Chinese subset",
        source="Hugging Face amphion/Emilia-Dataset: Emilia/ZH/",
        kind="hf",
        repo_id="amphion/Emilia-Dataset",
        hf_prefixes=("Emilia/ZH/",),
    ),
    "emilia-en": DatasetSpec(
        key="emilia-en",
        name="Emilia English subset",
        source="Hugging Face amphion/Emilia-Dataset: Emilia/EN/",
        kind="hf",
        repo_id="amphion/Emilia-Dataset",
        hf_prefixes=("Emilia/EN/",),
    ),
    "gigaspeech": DatasetSpec(
        key="gigaspeech",
        name="GigaSpeech",
        source="Hugging Face speechcolab/gigaspeech / GitHub SpeechColab/GigaSpeech",
        kind="hf",
        repo_id="speechcolab/gigaspeech",
        requires_auth=True,
        requires_filter=True,
        manual_message=(
            "GigaSpeech is gated. Accept the dataset terms on Hugging Face and complete "
            "the SpeechColab request form before downloading. Use --hf-prefix or "
            "--hf-include to choose a subset."
        ),
    ),
    "voxbox": DatasetSpec(
        key="voxbox",
        name="VoxBox",
        source="Hugging Face SparkAudio/voxbox",
        kind="hf",
        repo_id="SparkAudio/voxbox",
        requires_filter=True,
        manual_message=(
            "VoxBox is large. Pass --hf-prefix or --hf-include to select a subset, "
            "for example audios/commonvoice_cn/."
        ),
    ),
    "wenetspeech": DatasetSpec(
        key="wenetspeech",
        name="WenetSpeech",
        source="OpenSLR SLR121 / https://wenet.org.cn/WenetSpeech/",
        kind="wenetspeech",
        manual_message=(
            "WenetSpeech requires the official download password. Apply on the "
            "WenetSpeech website, then set WENETSPEECH_PASSWORD or "
            "WENETSPEECH_PASSWORD_FILE before running this downloader."
        ),
    ),
    "commonvoice-cn": DatasetSpec(
        key="commonvoice-cn",
        name="Common Voice Chinese",
        source="Hugging Face SparkAudio/voxbox: commonvoice_cn",
        kind="hf",
        repo_id="SparkAudio/voxbox",
        hf_prefixes=("audios/commonvoice_cn/", "metadata/"),
        hf_includes=(r"(?i)(^audios/commonvoice_cn/|^metadata/commonvoice_cn\.jsonl$)",),
        manual_message=(
            "For this reproduction profile, Common Voice Chinese is downloaded from "
            "the public VoxBox mirror. Official Mozilla releases now require Mozilla "
            "Data Collective access."
        ),
    ),
    "commonvoice-en": DatasetSpec(
        key="commonvoice-en",
        name="Common Voice English",
        source="Hugging Face SparkAudio/voxbox: commonvoice_en",
        kind="hf",
        repo_id="SparkAudio/voxbox",
        hf_prefixes=("audios/commonvoice_en/", "metadata/"),
        hf_includes=(r"(?i)(^audios/commonvoice_en/|^metadata/commonvoice_en\.jsonl$)",),
        manual_message=(
            "For this reproduction profile, Common Voice English is downloaded from "
            "the public VoxBox mirror. Official Mozilla releases now require Mozilla "
            "Data Collective access."
        ),
    ),
}


class DownloadError(RuntimeError):
    pass


def configure_log_file(path: Path | None) -> None:
    global LOG_FILE
    if path is None:
        return
    LOG_FILE = path.expanduser().resolve()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp()}] [log] start download_asr_datasets.py\n")


def _same_path(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return False


def _write_log_file_line(message: str) -> None:
    if LOG_FILE is None:
        return
    with LOG_FILE.open("a", encoding="utf-8") as f:
        lines = str(message).splitlines() or [""]
        for line in lines:
            f.write(f"[{timestamp()}] {line}\n")


def timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
    if LOG_FILE is None or _same_path(LOG_FILE, os.environ.get("SOULX_TEE_LOG_FILE")):
        return
    _write_log_file_line(message)


def tool_path(name: str) -> str | None:
    return shutil.which(name)


def tail_text(text: str, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    return "\n".join(lines[-max_lines:])


def detect_failure_reason(text: str, *, hf_context: bool = False) -> str:
    lowered = text.lower()
    if "no space left" in lowered or "disk quota" in lowered:
        return "磁盘空间不足，请清理目标盘或更换 --out-dir"
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return "请求被限流，请降低并发、稍后重试，或使用已登录的 Hugging Face token"
    if (
        "503" in lowered
        or "502" in lowered
        or "504" in lowered
        or "service unavailable" in lowered
        or "bad gateway" in lowered
        or "gateway timeout" in lowered
    ):
        return "远端服务或代理临时不可用，请重试；Hugging Face 下载建议启用 --turbo on"
    if "401" in lowered or "unauthorized" in lowered or "invalid token" in lowered:
        return "认证失败，请检查 HF_TOKEN 或 hf auth login 状态"
    if "403" in lowered or "forbidden" in lowered:
        if hf_context or "gated" in lowered or "restricted" in lowered:
            return "权限不足或 gated 数据集未授权，请先在 Hugging Face 网页接受条款并使用有 read 权限的 token"
        return "服务器拒绝访问，可能是镜像权限、限流或链接失效"
    if "404" in lowered or "not found" in lowered or "resource not found" in lowered:
        return "资源不存在，请检查数据集路径、revision、文件筛选条件或镜像链接"
    if "timed out" in lowered or "timeout" in lowered:
        return "连接或读取超时，可重试、启用 --turbo on，或调大 --timeout"
    if (
        "could not resolve" in lowered
        or "name resolution" in lowered
        or "temporary failure in name resolution" in lowered
    ):
        return "DNS 解析失败，请检查网络或启用 --turbo on"
    if (
        "connection refused" in lowered
        or "connection reset" in lowered
        or "network is unreachable" in lowered
        or "failed to establish" in lowered
    ):
        return "网络连接失败，请检查网络、代理或启用 --turbo on"
    if "eof occurred in violation of protocol" in lowered:
        return "TLS 连接被中途断开，通常是服务器网络、代理或 Hugging Face 连接不稳定；请重试或启用 --turbo on"
    if "certificate" in lowered or "ssl" in lowered or "tls" in lowered:
        return "TLS/证书校验失败，可能是镜像证书或代理问题"
    if "gated repo" in lowered or "gated dataset" in lowered or "restricted repo" in lowered:
        return "Hugging Face gated 数据集未授权，请先登录网页接受条款"
    if "repository not found" in lowered or "repo not found" in lowered:
        return "Hugging Face 仓库不存在，或当前 token 没有访问该仓库的权限"
    if "cannot find the requested files" in lowered or "no files" in lowered:
        return "文件筛选条件没有匹配结果，请检查 --hf-prefix、--hf-include 或 --hf-exclude"
    return "未分类错误，请查看下方工具输出"


def explain_process_failure(
    tool: str,
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    hf_context: bool = False,
) -> str:
    output = "\n".join(part for part in (stdout, stderr) if part)
    reason = detect_failure_reason(output, hf_context=hf_context)
    return f"{tool} 退出码 {returncode}：{reason}\n{tail_text(output)}"


def explain_exception(exc: BaseException, *, hf_context: bool = False) -> str:
    if isinstance(exc, DownloadError):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
        reason = detect_failure_reason(f"{exc.code} {exc.reason}\n{detail}", hf_context=hf_context)
        return f"HTTP {exc.code} {exc.reason}：{reason}"
    if isinstance(exc, urllib.error.URLError):
        detail = str(exc.reason)
        reason = detect_failure_reason(detail, hf_context=hf_context)
        return f"URL 访问失败：{reason}；底层原因：{detail}"
    if isinstance(exc, TimeoutError):
        return "请求超时，可重试、启用 --turbo on，或调大 --timeout"
    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC:
            return "磁盘空间不足，请清理目标盘或更换 --out-dir"
        return f"系统错误：{exc}"
    return str(exc)


def chunked(items: list[FileSpec], chunk_size: int) -> Iterable[list[FileSpec]]:
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


class TurboLoader:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.loaded = False

    def load_if_forced(self) -> None:
        if self.mode == "on":
            self.load()

    def load_for_hf_if_auto(self) -> None:
        if self.mode == "auto" and not self.loaded:
            self.load()

    def load_after_failure(self) -> None:
        if self.mode == "auto" and not self.loaded:
            self.load()

    def load(self) -> None:
        if self.loaded:
            return
        if not NETWORK_TURBO.exists():
            log(f"[turbo] {NETWORK_TURBO} not found; continuing without it.")
            self.loaded = True
            return

        command = f"source {NETWORK_TURBO} >/dev/null 2>&1; env -0"
        result = subprocess.run(
            ["bash", "-lc", command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", "replace").strip()
            raise DownloadError(f"failed to source {NETWORK_TURBO}: {stderr}")

        for item in result.stdout.split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            os.environ[key.decode("utf-8", "surrogateescape")] = value.decode(
                "utf-8", "surrogateescape"
            )

        self.loaded = True
        log(f"[turbo] sourced {NETWORK_TURBO}")


class Progress:
    def __init__(self, label: str, total: int | None, start: int = 0) -> None:
        self.label = label
        self.total = total
        self.start = start
        self.current = start
        self.started_at = time.monotonic()
        self.last_print = 0.0
        self.last_log = 0.0

    def update(self, downloaded: int, force: bool = False) -> None:
        self.current = downloaded
        now = time.monotonic()
        if force or now - self.last_log >= 60.0:
            self.last_log = now
            log(f"[progress] {self.status_text(now)}")
        if not force and now - self.last_print < 0.5:
            return
        self.last_print = now

        message = "\r" + self.status_text(now)
        print(message, end="", file=sys.stderr, flush=True)

    def status_text(self, now: float | None = None) -> str:
        if now is None:
            now = time.monotonic()
        elapsed = max(now - self.started_at, 1e-6)
        delta = max(self.current - self.start, 0)
        speed = delta / elapsed

        if self.total:
            percent = min(self.current / self.total * 100, 100.0)
            remaining = max(self.total - self.current, 0)
            eta = remaining / speed if speed > 0 else None
            return (
                f"{self.label}: {format_size(self.current)}/{format_size(self.total)} "
                f"({percent:6.2f}%) {format_size(speed)}/s ETA {format_seconds(eta)}"
            )
        return f"{self.label}: {format_size(self.current)} {format_size(speed)}/s"

    def finish(self) -> None:
        self.update(self.current, force=True)
        print(file=sys.stderr, flush=True)


def format_size(size: float | None) -> str:
    if size is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ASR datasets with resume support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List supported datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="aishell1")
    parser.add_argument("--profile", type=Path, help="Dataset profile YAML for multi-dataset download planning.")
    parser.add_argument("--plan-out", type=Path, help="Write the resolved multi-dataset download plan as JSON.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--files",
        default="all",
        help="Comma-separated file keys for direct HTTP datasets.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded tar archives. Archives are removed after successful extraction unless --keep-archives is set.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep tar archives after successful extraction.",
    )
    parser.add_argument(
        "--turbo",
        choices=("auto", "on", "off"),
        default="auto",
        help="Source /etc/network_turbo when needed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    parser.add_argument(
        "--backend",
        choices=("auto", "python", "aria2", "hf"),
        default="auto",
        help="Download backend. auto prefers aria2c for HTTP and hf for Hugging Face.",
    )
    parser.add_argument("--retries", type=int, default=5, help="Retry rounds per file.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Read chunk size in bytes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove wrong-size final files or oversized partials before downloading again.",
    )
    parser.add_argument("--revision", default=None, help="Hugging Face revision override.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to env token.")
    parser.add_argument(
        "--hf-prefix",
        action="append",
        default=[],
        help="Hugging Face path prefix to include. Can be repeated.",
    )
    parser.add_argument(
        "--hf-include",
        action="append",
        default=[],
        help="Regex for Hugging Face file paths to include. Can be repeated.",
    )
    parser.add_argument(
        "--hf-exclude",
        action="append",
        default=[],
        help="Regex for Hugging Face file paths to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--hf-all",
        action="store_true",
        help="Allow downloading an HF dataset without prefix/include filters.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit selected HF files.")
    parser.add_argument(
        "--hf-max-workers",
        type=int,
        default=8,
        help="Maximum workers for hf download.",
    )
    parser.add_argument(
        "--wenetspeech-password-file",
        type=Path,
        help="File containing the WenetSpeech official download password. Defaults to WENETSPEECH_PASSWORD_FILE or WENETSPEECH_PASSWORD.",
    )
    parser.add_argument(
        "--wenetspeech-toolkit-dir",
        type=Path,
        help="Existing or cloned WenetSpeech official toolkit directory.",
    )
    parser.add_argument(
        "--keep-wenetspeech-toolkit",
        action="store_true",
        help="Keep the cloned WenetSpeech toolkit after download for debugging.",
    )
    parser.add_argument(
        "--wenetspeech-download-dir",
        type=Path,
        help="Directory for encrypted WenetSpeech archives. Defaults to <target>/download.",
    )
    parser.add_argument(
        "--wenetspeech-untar-dir",
        type=Path,
        help="Directory for extracted WenetSpeech data. Defaults to <target>/extracted.",
    )
    parser.add_argument("--log-file", type=Path, help="Append download logs to this file.")
    return parser.parse_args()


def list_datasets() -> None:
    print("Supported datasets:")
    for key in sorted(DATASETS):
        dataset = DATASETS[key]
        extra = ""
        if dataset.kind == "http":
            extra = " files: " + ", ".join(file_spec.key for file_spec in dataset.files)
        elif dataset.kind == "hf":
            defaults = ", ".join(dataset.hf_prefixes) if dataset.hf_prefixes else "none"
            extra = f" repo: {dataset.repo_id}, default prefixes: {defaults}"
        elif dataset.kind == "wenetspeech":
            extra = " official toolkit; requires WENETSPEECH_PASSWORD"
        print(f"- {key}: {dataset.name} [{dataset.kind}]{extra}")


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def load_profile_tasks(profile_path: Path) -> list[DownloadTask]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise DownloadError("profile mode requires PyYAML") from exc
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    entries = data.get("datasets") or []
    if not isinstance(entries, list):
        raise DownloadError(f"profile datasets must be a list: {profile_path}")

    tasks: list[DownloadTask] = []
    for entry in entries:
        if isinstance(entry, str):
            logical_id = entry
            download_cfg: dict = {}
            enabled = True
        elif isinstance(entry, dict):
            logical_id = str(entry.get("id") or entry.get("dataset") or entry.get("name") or "")
            enabled = bool(entry.get("enabled", True))
            download_cfg = dict(entry.get("download") or {})
        else:
            raise DownloadError(f"invalid dataset profile entry: {entry!r}")
        if not logical_id or not enabled or bool(download_cfg.get("enabled", True)) is False:
            continue
        source_dataset = str(download_cfg.get("source_dataset") or logical_id)
        if source_dataset not in DATASETS:
            raise DownloadError(f"profile entry {logical_id} references unsupported source_dataset={source_dataset}")
        files_value = download_cfg.get("files", "all")
        if isinstance(files_value, list):
            files = ",".join(str(item) for item in files_value)
        else:
            files = str(files_value)
        dedup_key = str(
            download_cfg.get("dedup_key")
            or f"{source_dataset}:{files}:{download_cfg.get('revision') or DATASETS[source_dataset].revision}:"
            f"{','.join(_as_tuple(download_cfg.get('hf_prefix')))}:"
            f"{','.join(_as_tuple(download_cfg.get('hf_include')))}"
        )
        tasks.append(
            DownloadTask(
                logical_id=logical_id,
                source_dataset=source_dataset,
                dedup_key=dedup_key,
                files=files,
                revision=str(download_cfg["revision"]) if download_cfg.get("revision") is not None else None,
                hf_prefix=_as_tuple(download_cfg.get("hf_prefix")),
                hf_include=_as_tuple(download_cfg.get("hf_include")),
                hf_exclude=_as_tuple(download_cfg.get("hf_exclude")),
                hf_all=bool(download_cfg.get("hf_all", False)),
                max_files=int(download_cfg["max_files"]) if download_cfg.get("max_files") is not None else None,
            )
        )
    if not tasks:
        raise DownloadError(f"profile has no enabled download tasks: {profile_path}")
    return tasks


def deduplicate_tasks(tasks: list[DownloadTask]) -> tuple[list[DownloadTask], dict[str, list[str]]]:
    selected: list[DownloadTask] = []
    logical_ids_by_key: dict[str, list[str]] = {}
    seen: set[str] = set()
    for task in tasks:
        logical_ids_by_key.setdefault(task.dedup_key, []).append(task.logical_id)
        if task.dedup_key in seen:
            continue
        seen.add(task.dedup_key)
        selected.append(task)
    return selected, logical_ids_by_key


def args_for_task(args: argparse.Namespace, task: DownloadTask) -> argparse.Namespace:
    task_args = copy.copy(args)
    task_args.dataset = task.source_dataset
    task_args.files = task.files
    task_args.revision = task.revision
    task_args.hf_prefix = list(task.hf_prefix)
    task_args.hf_include = list(task.hf_include)
    task_args.hf_exclude = list(task.hf_exclude)
    task_args.hf_all = task.hf_all
    task_args.max_files = task.max_files
    return task_args


def task_to_plan_item(task: DownloadTask, dataset_dir: Path, logical_ids: list[str]) -> dict:
    dataset = DATASETS[task.source_dataset]
    return {
        "logical_ids": logical_ids,
        "source_dataset": task.source_dataset,
        "source_kind": dataset.kind,
        "dedup_key": task.dedup_key,
        "target_dir": str(dataset_dir),
        "files": task.files,
        "revision": task.revision or dataset.revision,
        "hf_prefix": list(task.hf_prefix or dataset.hf_prefixes),
        "hf_include": list(task.hf_include or dataset.hf_includes),
        "hf_exclude": list(task.hf_exclude or dataset.hf_excludes),
        "hf_all": task.hf_all,
        "max_files": task.max_files,
        "requires_auth": dataset.requires_auth,
        "source": dataset.source,
    }


def selected_files(dataset: DatasetSpec, selection: str) -> list[FileSpec]:
    if selection.strip().lower() == "all":
        return list(dataset.files)

    by_key = {file_spec.key: file_spec for file_spec in dataset.files}
    requested = [part.strip() for part in selection.split(",") if part.strip()]
    if not requested:
        raise DownloadError("--files cannot be empty")

    unknown = sorted(set(requested) - set(by_key))
    if unknown:
        valid = ", ".join(["all"] + sorted(by_key))
        raise DownloadError(f"unknown file key(s): {', '.join(unknown)}. Valid values: {valid}")

    return [by_key[key] for key in requested]


def load_state(dataset_dir: Path) -> dict:
    path = dataset_dir / STATE_FILE
    if not path.exists():
        return {"files": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"files": {}}


def save_state(dataset_dir: Path, dataset: DatasetSpec, state: dict) -> None:
    state["dataset"] = dataset.key
    state["dataset_name"] = dataset.name
    state["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    state.setdefault("files", {})
    path = dataset_dir / STATE_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def describe_http_plan(
    dataset: DatasetSpec,
    files: Iterable[FileSpec],
    dataset_dir: Path,
    args: argparse.Namespace,
) -> None:
    print(f"Dataset: {dataset.name} ({dataset.key})")
    print(f"Kind:    {dataset.kind}")
    print(f"Backend: {args.backend}")
    print(f"Source:  {dataset.source}")
    print(f"Target:  {dataset_dir}")
    for file_spec in files:
        print()
        print(f"- {file_spec.key}: {file_spec.filename}")
        print(f"  Size: {format_size(file_spec.size_bytes)} / listed {file_spec.display_size}")
        print(f"  Desc: {file_spec.description}")
        for index, url in enumerate(file_spec.urls, start=1):
            print(f"  URL {index}: {url}")


def describe_hf_plan(dataset: DatasetSpec, dataset_dir: Path, args: argparse.Namespace) -> None:
    revision = args.revision or dataset.revision
    prefixes = tuple(args.hf_prefix) or dataset.hf_prefixes
    includes = tuple(args.hf_include) or dataset.hf_includes
    excludes = tuple(args.hf_exclude) or dataset.hf_excludes
    print(f"Dataset: {dataset.name} ({dataset.key})")
    print(f"Kind:    Hugging Face")
    print(f"Backend: {args.backend}")
    print(f"Source:  {dataset.source}")
    print(f"Repo:    {dataset.repo_id}@{revision}")
    print(f"Target:  {dataset_dir}")
    print(f"Prefix:  {', '.join(prefixes) if prefixes else '(none)'}")
    print(f"Include: {', '.join(includes) if includes else '(none)'}")
    print(f"Exclude: {', '.join(excludes) if excludes else '(none)'}")
    if dataset.requires_auth:
        print("Auth:    likely required; set HF_TOKEN or pass --hf-token")
    if dataset.requires_filter and not args.hf_all and not prefixes and not args.hf_include:
        print("Note:    this dataset requires --hf-prefix/--hf-include or --hf-all")


def describe_manual(dataset: DatasetSpec, dataset_dir: Path) -> None:
    print(f"Dataset: {dataset.name} ({dataset.key})")
    print(f"Kind:    manual")
    print(f"Source:  {dataset.source}")
    print(f"Target:  {dataset_dir}")
    print()
    print(dataset.manual_message)


def request(
    url: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> urllib.request.Request:
    all_headers = {"User-Agent": USER_AGENT}
    if headers:
        all_headers.update(headers)
    return urllib.request.Request(url, headers=all_headers, method=method)


def auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def state_entry_for(state: dict, file_key: str) -> dict | None:
    entry = state.get("files", {}).get(file_key)
    return entry if isinstance(entry, dict) else None


def validate_existing(
    final_path: Path,
    file_spec: FileSpec,
    force: bool,
    state_entry: dict | None,
) -> bool:
    if not final_path.exists():
        return False

    size = final_path.stat().st_size
    if file_spec.size_bytes is not None and size == file_spec.size_bytes:
        log(f"[skip] {final_path} already exists ({format_size(size)}).")
        return True

    if state_entry and state_entry.get("size_bytes") == size:
        log(f"[skip] {final_path} already exists and matches state ({format_size(size)}).")
        return True

    if file_spec.size_bytes is None and not force:
        log(
            f"[skip] {final_path} exists ({format_size(size)}); "
            "no exact expected size is configured."
        )
        return True

    if force:
        log(f"[force] removing existing file {final_path} ({format_size(size)}).")
        final_path.unlink()
        return False

    raise DownloadError(
        f"{final_path} exists but has wrong size: "
        f"{format_size(size)} != {format_size(file_spec.size_bytes)}. "
        "Use --force to remove it and download again."
    )


def parse_content_range(value: str) -> tuple[int, int, int | None] | None:
    try:
        units, range_part = value.strip().split(" ", 1)
        if units.lower() != "bytes":
            return None
        span, total_text = range_part.split("/", 1)
        start_text, end_text = span.split("-", 1)
        total = None if total_text == "*" else int(total_text)
        return int(start_text), int(end_text), total
    except (ValueError, AttributeError):
        return None


def finalize_part(
    part_path: Path,
    final_path: Path,
    file_spec: FileSpec,
    observed_total: int | None,
) -> None:
    size = part_path.stat().st_size
    expected = file_spec.size_bytes or observed_total
    if expected is not None and size != expected:
        raise DownloadError(
            f"incomplete download for {file_spec.filename}: "
            f"{format_size(size)} != {format_size(expected)}"
        )
    if size <= 0:
        raise DownloadError(f"downloaded file is empty: {file_spec.filename}")
    os.replace(part_path, final_path)
    log(f"[file_done] {file_spec.filename} -> {final_path.expanduser().resolve()} ({format_size(size)})")


def download_file(
    file_spec: FileSpec,
    dataset_dir: Path,
    timeout: int,
    chunk_size: int,
    retries: int,
    force: bool,
    turbo: TurboLoader,
    state_entry: dict | None,
    extra_headers: dict[str, str] | None = None,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    final_path = dataset_dir / file_spec.filename
    part_path = Path(f"{final_path}.part")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    log(
        f"[file_start] {file_spec.filename} -> {final_path.expanduser().resolve()}；"
        f"expected_size={format_size(file_spec.size_bytes)}"
    )

    if validate_existing(final_path, file_spec, force, state_entry):
        return final_path

    if part_path.exists():
        part_size = part_path.stat().st_size
        if file_spec.size_bytes is not None and part_size == file_spec.size_bytes:
            finalize_part(part_path, final_path, file_spec, observed_total=None)
            return final_path
        if file_spec.size_bytes is not None and part_size > file_spec.size_bytes:
            if force:
                log(f"[force] removing oversized partial {part_path}.")
                part_path.unlink()
            else:
                raise DownloadError(
                    f"{part_path} is larger than expected. Use --force to remove it."
                )

    last_error: Exception | None = None
    retry_rounds = max(retries, 1)

    for round_index in range(1, retry_rounds + 1):
        for url_index, url in enumerate(file_spec.urls, start=1):
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            log(
                f"[download] {file_spec.filename} round {round_index}/{retry_rounds}, "
                f"mirror {url_index}/{len(file_spec.urls)}"
            )
            try:
                observed_total = attempt_download(
                    url=url,
                    file_spec=file_spec,
                    part_path=part_path,
                    resume_from=resume_from,
                    timeout=timeout,
                    chunk_size=chunk_size,
                    extra_headers=extra_headers or {},
                )
                finalize_part(part_path, final_path, file_spec, observed_total)
                return final_path
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = exc
                log(f"[warn] {url} failed: {explain_exception(exc)}")
                turbo.load_after_failure()

    raise DownloadError(
        f"failed to download {file_spec.filename}: {explain_exception(last_error) if last_error else 'unknown error'}"
    )


def download_file_aria2(
    file_spec: FileSpec,
    dataset_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
    state_entry: dict | None,
) -> Path:
    if not tool_path("aria2c"):
        raise DownloadError("aria2c 未安装，无法使用 --backend aria2")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    final_path = dataset_dir / file_spec.filename
    part_path = Path(f"{final_path}.part")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    log(
        f"[file_start] {file_spec.filename} -> {final_path.expanduser().resolve()}；"
        f"expected_size={format_size(file_spec.size_bytes)}"
    )

    if validate_existing(final_path, file_spec, force, state_entry):
        return final_path

    if part_path.exists():
        part_size = part_path.stat().st_size
        if file_spec.size_bytes is not None and part_size == file_spec.size_bytes:
            finalize_part(part_path, final_path, file_spec, observed_total=None)
            return final_path
        if file_spec.size_bytes is not None and part_size > file_spec.size_bytes:
            if force:
                log(f"[force] removing oversized partial {part_path}.")
                part_path.unlink()
                control = Path(f"{part_path}.aria2")
                if control.exists():
                    control.unlink()
            else:
                raise DownloadError(
                    f"{part_path} is larger than expected. Use --force to remove it."
                )

    cmd = [
        "aria2c",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--file-allocation=none",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=1M",
        f"--max-tries={max(retries, 1)}",
        "--retry-wait=5",
        f"--timeout={timeout}",
        f"--connect-timeout={timeout}",
        "--summary-interval=5",
        "--show-console-readout=true",
        "--console-log-level=notice",
        "--download-result=hide",
        "--dir",
        str(part_path.parent),
        "--out",
        part_path.name,
        *file_spec.urls,
    ]
    log_path = part_path.with_name(f".{part_path.name}.aria2.log")
    cmd[1:1] = [f"--log={log_path}", "--log-level=notice"]
    log(f"[aria2] downloading {file_spec.filename} with {len(file_spec.urls)} mirror(s)")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log_text = ""
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        raise DownloadError(
            explain_process_failure("aria2c", result.returncode, "", log_text)
        )

    control = Path(f"{part_path}.aria2")
    if control.exists():
        raise DownloadError(f"aria2c ended but control file remains: {control}")
    if log_path.exists():
        log_path.unlink()
    finalize_part(part_path, final_path, file_spec, observed_total=None)
    return final_path


def download_file_with_backend(
    file_spec: FileSpec,
    dataset_dir: Path,
    args: argparse.Namespace,
    turbo: TurboLoader,
    state_entry: dict | None,
) -> Path:
    if args.backend == "hf":
        raise DownloadError("--backend hf 仅适用于 Hugging Face 数据集；HTTP 数据集请使用 auto/python/aria2")

    if args.backend in {"auto", "aria2"}:
        if tool_path("aria2c"):
            try:
                return download_file_aria2(
                    file_spec=file_spec,
                    dataset_dir=dataset_dir,
                    timeout=args.timeout,
                    retries=args.retries,
                    force=args.force,
                    state_entry=state_entry,
                )
            except DownloadError as exc:
                if args.backend == "aria2":
                    raise
                log(f"[warn] aria2c failed; falling back to Python downloader: {exc}")
                turbo.load_after_failure()
        elif args.backend == "aria2":
            raise DownloadError("aria2c 未安装，无法使用 --backend aria2")

    return download_file(
        file_spec=file_spec,
        dataset_dir=dataset_dir,
        timeout=args.timeout,
        chunk_size=args.chunk_size,
        retries=args.retries,
        force=args.force,
        turbo=turbo,
        state_entry=state_entry,
    )


def attempt_download(
    url: str,
    file_spec: FileSpec,
    part_path: Path,
    resume_from: int,
    timeout: int,
    chunk_size: int,
    extra_headers: dict[str, str],
) -> int | None:
    headers = dict(extra_headers)
    mode = "wb"
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    req = request(url, headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        observed_total = file_spec.size_bytes
        if resume_from and status == 206:
            content_range = parse_content_range(response.headers.get("Content-Range", ""))
            if not content_range or content_range[0] != resume_from:
                raise DownloadError(
                    f"unexpected Content-Range while resuming: "
                    f"{response.headers.get('Content-Range')!r}"
                )
            observed_total = observed_total or content_range[2]
        elif resume_from and status == 200:
            log("[resume] server ignored Range; restarting this file.")
            resume_from = 0
            mode = "wb"
            content_length = response.headers.get("Content-Length")
            if observed_total is None and content_length:
                observed_total = int(content_length)
        elif status not in (200, 206):
            raise DownloadError(f"unexpected HTTP status {status}")
        else:
            content_length = response.headers.get("Content-Length")
            if observed_total is None and content_length:
                observed_total = int(content_length)

        progress = Progress(file_spec.filename, total=observed_total, start=resume_from)
        downloaded = resume_from
        progress.update(downloaded, force=True)

        with part_path.open(mode) as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                progress.update(downloaded)
            output.flush()
            os.fsync(output.fileno())
        progress.current = downloaded
        progress.finish()
        return observed_total


def is_tar_like(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2"))


def remove_archive_after_extract(archive_path: Path) -> None:
    if not archive_path.exists():
        return
    archive_path.unlink()
    log(f"[cleanup] removed archive {archive_path}")


def extract_archive(archive_path: Path, extract_dir: Path, delete_archive: bool) -> None:
    if not is_tar_like(archive_path):
        log(f"[skip] {archive_path.name} is not a tar archive.")
        return

    marker = extract_dir / f".{archive_path.name}.done"
    if marker.exists():
        log(f"[skip] {archive_path.name} already extracted.")
        if delete_archive:
            remove_archive_after_extract(archive_path)
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    log(f"[extract] {archive_path} -> {extract_dir}")
    with tarfile.open(archive_path, mode="r:*") as archive:
        safe_extract(archive, extract_dir)
    marker.write_text(
        _dt.datetime.now(_dt.timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )
    log(f"[extract_done] {archive_path.name} -> {extract_dir.expanduser().resolve()}")
    if delete_archive:
        remove_archive_after_extract(archive_path)


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    members = archive.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        if os.path.commonpath([str(destination), str(target)]) != str(destination):
            raise DownloadError(f"archive member would escape target directory: {member.name}")
    archive.extractall(destination, members=members)


def quote_repo_id(repo_id: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))


def quote_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))


def hf_tree_url(repo_id: str, revision: str, path: str = "") -> str:
    repo = quote_repo_id(repo_id)
    rev = urllib.parse.quote(revision, safe="")
    suffix = f"/{quote_path(path.strip('/'))}" if path.strip("/") else ""
    return f"{HF_ENDPOINT}/api/datasets/{repo}/tree/{rev}{suffix}?recursive=1&expand=1"


def hf_resolve_url(repo_id: str, revision: str, path: str) -> str:
    repo = quote_repo_id(repo_id)
    rev = urllib.parse.quote(revision, safe="")
    return f"{HF_ENDPOINT}/datasets/{repo}/resolve/{rev}/{quote_path(path)}"


def parse_next_link(headers: object) -> str | None:
    link = headers.get("Link") if hasattr(headers, "get") else None
    if not link:
        return None
    for part in link.split(","):
        sections = [section.strip() for section in part.split(";")]
        if len(sections) >= 2 and sections[1] == 'rel="next"':
            return sections[0].strip("<>")
    return None


def is_retryable_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in {408, 429, 500, 502, 503, 504}


def read_json_url(
    url: str,
    timeout: int,
    headers: dict[str, str],
    *,
    retries: int,
    turbo: TurboLoader,
    label: str,
) -> tuple[object, str | None]:
    attempts = max(retries, 1)
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        log(f"[hf_list] {label} attempt {attempt}/{attempts}")
        try:
            req = request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data, parse_next_link(response.headers)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not is_retryable_http_error(exc) or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == attempts:
                raise

        log(f"[hf_list_warn] {label} failed: {explain_exception(last_error, hf_context=True)}")
        turbo.load_after_failure()
        time.sleep(min(30, 2 ** min(attempt, 5)))

    if last_error is not None:
        raise last_error
    raise DownloadError(f"failed to read Hugging Face API: {label}")


def compile_patterns(patterns: Iterable[str], label: str) -> list[re.Pattern[str]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise DownloadError(f"invalid {label} regex {pattern!r}: {exc}") from exc
    return compiled


def hf_query_path_for_prefix(prefix: str) -> str:
    clean = prefix.strip("/")
    if not clean:
        return ""
    name = clean.rsplit("/", 1)[-1]
    if "." in name and "/" in clean:
        return clean.rsplit("/", 1)[0]
    if "." in name:
        return ""
    return clean


def list_hf_files(
    dataset: DatasetSpec,
    args: argparse.Namespace,
    headers: dict[str, str],
    turbo: TurboLoader,
) -> list[FileSpec]:
    if not dataset.repo_id:
        raise DownloadError(f"{dataset.key} has no Hugging Face repo configured")

    revision = args.revision or dataset.revision
    prefixes = tuple(args.hf_prefix) or dataset.hf_prefixes
    if dataset.requires_filter and not args.hf_all and not prefixes and not args.hf_include:
        raise DownloadError(dataset.manual_message)

    includes = tuple(args.hf_include) or dataset.hf_includes
    excludes = tuple(args.hf_exclude) or dataset.hf_excludes
    include_patterns = compile_patterns(includes, "--hf-include")
    exclude_patterns = compile_patterns(excludes, "--hf-exclude")

    selected: list[FileSpec] = []
    seen: set[str] = set()
    query_paths = [""]
    if prefixes and not args.hf_all:
        query_paths = sorted({hf_query_path_for_prefix(prefix) for prefix in prefixes})

    for query_path in query_paths:
        url = hf_tree_url(dataset.repo_id, revision, query_path)
        page_index = 0
        while url:
            page_index += 1
            label = f"{dataset.repo_id}@{revision}:{query_path or '/'} page {page_index}"
            payload, url = read_json_url(
                url,
                timeout=args.timeout,
                headers=headers,
                retries=args.retries,
                turbo=turbo,
                label=label,
            )
            if isinstance(payload, dict) and payload.get("error"):
                raise DownloadError(str(payload["error"]))
            if not isinstance(payload, list):
                raise DownloadError(f"unexpected Hugging Face API response: {type(payload).__name__}")

            for item in payload:
                if not isinstance(item, dict) or item.get("type") != "file":
                    continue
                path = item.get("path")
                if not isinstance(path, str):
                    continue
                if path in {"README.md", ".gitattributes"} and not args.hf_all:
                    continue
                if prefixes and not any(path.startswith(prefix.strip("/")) for prefix in prefixes):
                    continue
                if include_patterns and not any(pattern.search(path) for pattern in include_patterns):
                    continue
                if exclude_patterns and any(pattern.search(path) for pattern in exclude_patterns):
                    continue
                if path in seen:
                    continue
                seen.add(path)

                size = item.get("size")
                size_bytes = size if isinstance(size, int) else None
                selected.append(
                    FileSpec(
                        key=path,
                        filename=path,
                        description=f"{dataset.repo_id}:{path}",
                        urls=(hf_resolve_url(dataset.repo_id, revision, path),),
                        size_bytes=size_bytes,
                        display_size=format_size(size_bytes),
                    )
                )
        log(
            f"[hf_list_done] {dataset.repo_id}@{revision}:{query_path or '/'}；"
            f"matched_so_far={len(selected)}"
        )

    selected.sort(key=lambda file_spec: file_spec.filename)
    if args.max_files is not None:
        selected = selected[: args.max_files]
    if not selected:
        raise DownloadError("no Hugging Face files matched the requested filters")
    return selected


def has_hf_token(args: argparse.Namespace) -> bool:
    return bool(args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


def extracted_marker_for(dataset_dir: Path, file_spec: FileSpec) -> Path:
    return dataset_dir / "extracted" / f".{Path(file_spec.filename).name}.done"


def can_skip_download_for_extracted_archive(
    dataset_dir: Path,
    file_spec: FileSpec,
    args: argparse.Namespace,
    state_entry: dict | None,
) -> bool:
    if not args.extract or args.keep_archives or not state_entry:
        return False
    if not is_tar_like(Path(file_spec.filename)):
        return False
    marker = extracted_marker_for(dataset_dir, file_spec)
    if marker.exists():
        log(
            f"[skip] {file_spec.filename} already extracted; "
            "archive was removed after extraction."
        )
        return True
    return False


def write_state_entry(state: dict, file_spec: FileSpec, final_path: Path) -> None:
    state.setdefault("files", {})[file_spec.key] = {
        "filename": file_spec.filename,
        "size_bytes": final_path.stat().st_size,
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "urls": list(file_spec.urls),
    }


def download_hf_files_with_cli(
    dataset: DatasetSpec,
    files: list[FileSpec],
    dataset_dir: Path,
    args: argparse.Namespace,
    state: dict,
) -> list[Path]:
    if not dataset.repo_id:
        raise DownloadError(f"{dataset.key} has no Hugging Face repo configured")
    if not tool_path("hf"):
        raise DownloadError("hf CLI 未安装，无法使用 --backend hf")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    pending: list[FileSpec] = []
    downloaded: list[Path] = []
    for file_spec in files:
        final_path = dataset_dir / file_spec.filename
        final_path.parent.mkdir(parents=True, exist_ok=True)
        log(
            f"[file_start] {file_spec.filename} -> {final_path.expanduser().resolve()}；"
            f"expected_size={format_size(file_spec.size_bytes)}"
        )
        state_entry = state_entry_for(state, file_spec.key)
        if can_skip_download_for_extracted_archive(dataset_dir, file_spec, args, state_entry):
            downloaded.append(final_path)
            continue
        if validate_existing(final_path, file_spec, args.force, state_entry):
            downloaded.append(final_path)
            continue
        pending.append(file_spec)

    if not pending:
        return downloaded

    revision = args.revision or dataset.revision
    env = os.environ.copy()
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if dataset.requires_auth and not has_hf_token(args):
        log(
            "[hf] no HF_TOKEN detected. If this is a gated dataset, run "
            "`export HF_TOKEN=...` or `hf auth login` after accepting the dataset terms."
        )

    log(f"[hf] downloading {len(pending)} file(s) with hf CLI")
    for batch in chunked(pending, 100):
        cmd = [
            "hf",
            "download",
            dataset.repo_id,
            *[file_spec.filename for file_spec in batch],
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(dataset_dir),
            "--max-workers",
            str(args.hf_max_workers),
        ]
        if args.force:
            cmd.append("--force-download")

        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            sample = ", ".join(file_spec.filename for file_spec in batch[:3])
            if len(batch) > 3:
                sample += f", ... ({len(batch)} files)"
            raise DownloadError(
                f"hf download failed for {dataset.repo_id}@{revision} [{sample}]\n"
                + explain_process_failure(
                    "hf download",
                    result.returncode,
                    result.stdout,
                    result.stderr,
                    hf_context=True,
                )
            )

        for file_spec in batch:
            final_path = dataset_dir / file_spec.filename
            if not final_path.exists():
                raise DownloadError(
                    f"hf download completed but expected file is missing: {final_path}"
                )
            size = final_path.stat().st_size
            if file_spec.size_bytes is not None and size != file_spec.size_bytes:
                raise DownloadError(
                    f"hf download size mismatch for {file_spec.filename}: "
                    f"{format_size(size)} != {format_size(file_spec.size_bytes)}"
                )
            write_state_entry(state, file_spec, final_path)
            save_state(dataset_dir, dataset, state)
            downloaded.append(final_path)
            log(f"[file_done] {file_spec.filename} -> {final_path.expanduser().resolve()} ({format_size(size)})")

    return downloaded


def download_http_dataset(
    dataset: DatasetSpec,
    dataset_dir: Path,
    args: argparse.Namespace,
    turbo: TurboLoader,
) -> list[Path]:
    files = selected_files(dataset, args.files)
    if args.dry_run:
        describe_http_plan(dataset, files, dataset_dir, args)
        return []
    log(
        f"[dataset_start] {dataset.key} ({dataset.name})；kind=http；"
        f"target={dataset_dir.expanduser().resolve()}；files={len(files)}"
    )

    state = load_state(dataset_dir)
    downloaded: list[Path] = []
    for file_spec in files:
        log(
            f"[file_plan] {dataset.key}/{file_spec.key} -> "
            f"{(dataset_dir / file_spec.filename).expanduser().resolve()}；"
            f"expected_size={format_size(file_spec.size_bytes)}"
        )
        state_entry = state_entry_for(state, file_spec.key)
        if can_skip_download_for_extracted_archive(dataset_dir, file_spec, args, state_entry):
            downloaded.append(dataset_dir / file_spec.filename)
            continue
        final_path = download_file_with_backend(
            file_spec=file_spec,
            dataset_dir=dataset_dir,
            args=args,
            turbo=turbo,
            state_entry=state_entry,
        )
        write_state_entry(state, file_spec, final_path)
        save_state(dataset_dir, dataset, state)
        downloaded.append(final_path)
    log(
        f"[dataset_files_done] {dataset.key}；"
        f"target={dataset_dir.expanduser().resolve()}；files={len(downloaded)}"
    )
    return downloaded


def download_hf_dataset(
    dataset: DatasetSpec,
    dataset_dir: Path,
    args: argparse.Namespace,
    turbo: TurboLoader,
) -> list[Path]:
    if args.dry_run:
        describe_hf_plan(dataset, dataset_dir, args)
        return []
    if args.backend == "aria2":
        raise DownloadError("--backend aria2 仅适用于 HTTP/OpenSLR 数据集；Hugging Face 数据集请使用 auto/python/hf")

    turbo.load_for_hf_if_auto()
    headers = auth_headers(args.hf_token)
    files = list_hf_files(dataset, args, headers, turbo)
    log(
        f"[dataset_start] {dataset.key} ({dataset.name})；kind=hf；"
        f"target={dataset_dir.expanduser().resolve()}；selected_files={len(files)}"
    )
    log(f"[hf] selected {len(files)} file(s)")

    state = load_state(dataset_dir)
    if args.backend in {"auto", "hf"}:
        if tool_path("hf"):
            try:
                return download_hf_files_with_cli(dataset, files, dataset_dir, args, state)
            except DownloadError as exc:
                if args.backend == "hf":
                    raise
                log(f"[warn] hf CLI failed; falling back to Python downloader: {exc}")
        elif args.backend == "hf":
            raise DownloadError("hf CLI 未安装，无法使用 --backend hf")

    downloaded: list[Path] = []
    for file_spec in files:
        final_path = download_file(
            file_spec=file_spec,
            dataset_dir=dataset_dir,
            timeout=args.timeout,
            chunk_size=args.chunk_size,
            retries=args.retries,
            force=args.force,
            turbo=turbo,
            state_entry=state_entry_for(state, file_spec.key),
            extra_headers=headers,
        )
        write_state_entry(state, file_spec, final_path)
        save_state(dataset_dir, dataset, state)
        downloaded.append(final_path)
    log(
        f"[dataset_files_done] {dataset.key}；"
        f"target={dataset_dir.expanduser().resolve()}；files={len(downloaded)}"
    )
    return downloaded


def describe_wenetspeech_plan(dataset: DatasetSpec, dataset_dir: Path, args: argparse.Namespace) -> None:
    toolkit_dir = args.wenetspeech_toolkit_dir or dataset_dir / "_toolkit" / "WenetSpeech"
    download_dir = args.wenetspeech_download_dir or dataset_dir / "download"
    untar_dir = args.wenetspeech_untar_dir or dataset_dir / "extracted"
    print(f"Dataset: {dataset.name} ({dataset.key})")
    print(f"Kind:    WenetSpeech official toolkit")
    print(f"Source:  {dataset.source}")
    print(f"Toolkit: {toolkit_dir}")
    print(f"Download:{download_dir}")
    print(f"Extract: {untar_dir}")
    print()
    print(dataset.manual_message)


def read_wenetspeech_password(args: argparse.Namespace) -> str:
    password_file = args.wenetspeech_password_file
    env_file = os.environ.get("WENETSPEECH_PASSWORD_FILE")
    if password_file is None and env_file:
        password_file = Path(env_file)
    if password_file is not None:
        try:
            password = password_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DownloadError(f"failed to read WenetSpeech password file {password_file}: {exc}") from exc
        if password:
            return password
        raise DownloadError(f"WenetSpeech password file is empty: {password_file}")

    password = os.environ.get("WENETSPEECH_PASSWORD", "").strip()
    if password:
        return password

    raise DownloadError(
        "WenetSpeech 需要官方申请到的下载密码。请先在 WenetSpeech 官网申请，"
        "然后设置 WENETSPEECH_PASSWORD 或 WENETSPEECH_PASSWORD_FILE。"
    )


def run_logged_process(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    label: str,
) -> None:
    log(f"[{label}] running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            tail.append(text)
            tail = tail[-20:]
            log(f"[{label}] {text}")
    returncode = process.wait()
    if returncode != 0:
        raise DownloadError(
            f"{label} failed with exit code {returncode}\n" + "\n".join(tail[-12:])
        )


def ensure_wenetspeech_toolkit(dataset_dir: Path, args: argparse.Namespace) -> tuple[Path, bool]:
    toolkit_dir = (
        args.wenetspeech_toolkit_dir.expanduser().resolve()
        if args.wenetspeech_toolkit_dir
        else (dataset_dir / "_toolkit" / "WenetSpeech").resolve()
    )
    managed_toolkit = args.wenetspeech_toolkit_dir is None
    script = toolkit_dir / "utils" / "download_wenetspeech.sh"
    if script.exists():
        log(f"[wenetspeech] reuse toolkit: {toolkit_dir}")
        return toolkit_dir, managed_toolkit
    if toolkit_dir.exists():
        raise DownloadError(
            f"WenetSpeech toolkit directory exists but script is missing: {script}"
        )
    if not tool_path("git"):
        raise DownloadError("git 未安装，无法自动获取 WenetSpeech 官方 toolkit")

    toolkit_dir.parent.mkdir(parents=True, exist_ok=True)
    run_logged_process(
        ["git", "clone", "--depth", "1", WENETSPEECH_GIT_URL, str(toolkit_dir)],
        cwd=toolkit_dir.parent,
        label="wenetspeech_git",
    )
    if not script.exists():
        raise DownloadError(f"WenetSpeech toolkit clone completed but script is missing: {script}")
    return toolkit_dir, managed_toolkit


def cleanup_wenetspeech_toolkit(toolkit_dir: Path, password_path: Path, *, remove_toolkit: bool) -> None:
    try:
        if password_path.exists():
            password_path.unlink()
            log(f"[wenetspeech_cleanup] removed password file: {password_path}")
    except OSError as exc:
        log(f"[warn] failed to remove WenetSpeech password file {password_path}: {exc}")

    if not remove_toolkit:
        return
    try:
        if toolkit_dir.exists():
            shutil.rmtree(toolkit_dir)
            log(f"[wenetspeech_cleanup] removed toolkit: {toolkit_dir}")
        parent = toolkit_dir.parent
        if parent.name == "_toolkit" and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            log(f"[wenetspeech_cleanup] removed empty toolkit dir: {parent}")
    except OSError as exc:
        log(f"[warn] failed to remove WenetSpeech toolkit {toolkit_dir}: {exc}")


def download_wenetspeech_dataset(
    dataset: DatasetSpec,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    if args.dry_run:
        describe_wenetspeech_plan(dataset, dataset_dir, args)
        return []

    if not tool_path("wget"):
        raise DownloadError("wget 未安装，WenetSpeech 官方下载脚本需要 wget")
    if not tool_path("openssl"):
        raise DownloadError("openssl 未安装，WenetSpeech 官方下载脚本需要 openssl")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    password = read_wenetspeech_password(args)
    toolkit_dir, managed_toolkit = ensure_wenetspeech_toolkit(dataset_dir, args)
    download_dir = (
        args.wenetspeech_download_dir.expanduser().resolve()
        if args.wenetspeech_download_dir
        else (dataset_dir / "download").resolve()
    )
    untar_dir = (
        args.wenetspeech_untar_dir.expanduser().resolve()
        if args.wenetspeech_untar_dir
        else (dataset_dir / "extracted").resolve()
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    untar_dir.mkdir(parents=True, exist_ok=True)

    safebox = toolkit_dir / "SAFEBOX"
    safebox.mkdir(parents=True, exist_ok=True)
    password_path = safebox / "password"
    password_path.write_text(password + "\n", encoding="utf-8")
    os.chmod(password_path, 0o600)

    try:
        log(
            f"[dataset_start] {dataset.key} ({dataset.name})；kind=wenetspeech；"
            f"download={download_dir}；extract={untar_dir}"
        )
        run_logged_process(
            ["bash", "utils/download_wenetspeech.sh", str(download_dir), str(untar_dir)],
            cwd=toolkit_dir,
            label="wenetspeech",
        )
    finally:
        cleanup_wenetspeech_toolkit(
            toolkit_dir,
            password_path,
            remove_toolkit=managed_toolkit and not args.keep_wenetspeech_toolkit,
        )

    state = load_state(dataset_dir)
    state.setdefault("files", {})["official_toolkit"] = {
        "filename": str(untar_dir),
        "size_bytes": None,
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "urls": [WENETSPEECH_GIT_URL, dataset.source],
    }
    save_state(dataset_dir, dataset, state)
    log(f"[wenetspeech_done] extracted_dir={untar_dir}")
    return [untar_dir]


def maybe_extract(paths: Iterable[Path], dataset_dir: Path, keep_archives: bool) -> None:
    extract_dir = dataset_dir / "extracted"
    paths = list(paths)
    if paths:
        log(
            f"[extract_start] target={extract_dir.expanduser().resolve()}；"
            f"archives={len(paths)}；keep_archives={keep_archives}"
        )
    for path in paths:
        extract_archive(path, extract_dir, delete_archive=not keep_archives)


def log_dataset_success(
    dataset: DatasetSpec,
    dataset_dir: Path,
    downloaded: list[Path],
    *,
    extracted: bool,
) -> None:
    status = "下载并解压完成" if extracted else "下载完成"
    log(
        f"[dataset_success] 数据集 {dataset.key} {status}；"
        f"存放位置：{dataset_dir.expanduser().resolve()}；"
        f"文件数：{len(downloaded)}"
    )


def run_dataset_download(dataset: DatasetSpec, dataset_dir: Path, args: argparse.Namespace, turbo: TurboLoader) -> list[Path]:
    if dataset.kind == "manual":
        log(
            f"[dataset_manual] {dataset.key} ({dataset.name})；"
            f"target={dataset_dir.expanduser().resolve()}；message={dataset.manual_message}"
        )
        describe_manual(dataset, dataset_dir)
        return []
    if dataset.kind == "http":
        return download_http_dataset(dataset, dataset_dir, args, turbo)
    if dataset.kind == "hf":
        return download_hf_dataset(dataset, dataset_dir, args, turbo)
    if dataset.kind == "wenetspeech":
        return download_wenetspeech_dataset(dataset, dataset_dir, args)
    raise DownloadError(f"unsupported dataset kind: {dataset.kind}")


def run_profile(args: argparse.Namespace, turbo: TurboLoader) -> int:
    tasks, logical_ids_by_key = deduplicate_tasks(load_profile_tasks(args.profile))
    plan_items = []
    for task in tasks:
        dataset = DATASETS[task.source_dataset]
        dataset_dir = args.out_dir.expanduser().resolve() / dataset.key
        plan_items.append(task_to_plan_item(task, dataset_dir, logical_ids_by_key[task.dedup_key]))

    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps({"tasks": plan_items}, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[plan] wrote {args.plan_out}")

    if args.dry_run:
        log(f"[dry_run] profile={args.profile}；physical_tasks={len(tasks)}")
        print(f"Profile: {args.profile}")
        print(f"Physical download tasks: {len(tasks)}")
        print()
        for item in plan_items:
            log(
                f"[dry_run_task] source_dataset={item['source_dataset']}；"
                f"kind={item['source_kind']}；target={item['target_dir']}；"
                f"logical_ids={','.join(item['logical_ids'])}"
            )
            print(f"- source_dataset: {item['source_dataset']} [{item['source_kind']}]")
            print(f"  logical_ids: {', '.join(item['logical_ids'])}")
            print(f"  dedup_key: {item['dedup_key']}")
            print(f"  target_dir: {item['target_dir']}")
            print(f"  files: {item['files']}")
            if item["hf_prefix"]:
                print(f"  hf_prefix: {', '.join(item['hf_prefix'])}")
            if item["hf_include"]:
                print(f"  hf_include: {', '.join(item['hf_include'])}")
            if item["requires_auth"]:
                print("  auth: HF_TOKEN or prior login likely required")
            print()
        return 0

    for task in tasks:
        dataset = DATASETS[task.source_dataset]
        task_args = args_for_task(args, task)
        dataset_dir = task_args.out_dir.expanduser().resolve() / dataset.key
        log(
            f"[profile] {task.logical_id} -> {task.source_dataset} "
            f"(dedup_key={task.dedup_key})；target={dataset_dir}"
        )
        downloaded = run_dataset_download(dataset, dataset_dir, task_args, turbo)
        if task_args.extract and downloaded and dataset.kind in {"http", "hf"}:
            maybe_extract(downloaded, dataset_dir, keep_archives=task_args.keep_archives)
        if dataset.kind == "manual":
            log(f"[manual] 数据集 {dataset.key} 需要手动下载；目标位置：{dataset_dir.expanduser().resolve()}")
        else:
            log_dataset_success(
                dataset,
                dataset_dir,
                downloaded,
                extracted=bool(task_args.extract and downloaded),
            )
    return 0


def main() -> int:
    args = parse_args()
    configure_log_file(args.log_file)
    if args.list:
        list_datasets()
        return 0

    try:
        turbo = TurboLoader(args.turbo)
        turbo.load_if_forced()

        if args.profile:
            return run_profile(args, turbo)

        dataset = DATASETS[args.dataset]
        dataset_dir = args.out_dir.expanduser().resolve() / dataset.key
        log(f"[single] dataset={dataset.key}；target={dataset_dir}")
        if dataset.kind == "manual":
            log(
                f"[dataset_manual] {dataset.key} ({dataset.name})；"
                f"target={dataset_dir.expanduser().resolve()}；message={dataset.manual_message}"
            )
            describe_manual(dataset, dataset_dir)
            return 0 if args.dry_run else 2
        downloaded = run_dataset_download(dataset, dataset_dir, args, turbo)
        if args.extract and downloaded and dataset.kind in {"http", "hf"}:
            maybe_extract(downloaded, dataset_dir, keep_archives=args.keep_archives)
        if not args.dry_run:
            log_dataset_success(
                dataset,
                dataset_dir,
                downloaded,
                extracted=bool(args.extract and downloaded),
            )
        return 0
    except KeyboardInterrupt:
        log("\n[interrupt] download stopped; rerun the same command to resume.")
        return 130
    except (DownloadError, OSError, urllib.error.URLError) as exc:
        log(f"[error] {explain_exception(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
