#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/root/autodl-tmp/models"))
DEFAULT_CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", "/root/autodl-tmp/cache"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/outputs"))
STATE_FILE = ".soulx_model_assets_state.json"
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"


@dataclass(frozen=True)
class Asset:
    key: str
    label: str
    kind: str
    source: str
    target_name: str
    required: bool
    validator: Callable[[Path, Path], tuple[bool, str]]


def log(message: str) -> None:
    print(message, flush=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_state(model_root: Path) -> dict:
    path = model_root / STATE_FILE
    if not path.exists():
        return {"assets": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"assets": {}}


def save_state(model_root: Path, state: dict) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    path = model_root / STATE_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def run_stream(command: list[str], *, env: dict[str, str], cwd: Path | None = None) -> None:
    log("[cmd] " + " ".join(command))
    inherit_stdio = sys.stdout.isatty() and sys.stderr.isatty()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=None if inherit_stdio else subprocess.PIPE,
        stderr=None if inherit_stdio else subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        if inherit_stdio:
            returncode = process.wait()
        else:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
            returncode = process.wait()
    except KeyboardInterrupt:
        terminate_process_group(process)
        raise
    except BaseException:
        terminate_process_group(process)
        raise
    if returncode != 0:
        raise RuntimeError(f"command failed with exit code {returncode}: {' '.join(command)}")


def capture_command(command: list[str], *, env: dict[str, str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    log("[interrupt] terminating child process group")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        log("[interrupt] child process group did not exit; killing")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def any_file_with_suffix(root: Path, suffixes: tuple[str, ...]) -> bool:
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            return True
    return False


def has_any_file(root: Path, names: tuple[str, ...]) -> bool:
    return any((root / name).exists() for name in names)


def validate_qwen(target: Path, cache_root: Path) -> tuple[bool, str]:
    if not target.exists():
        return False, "target directory is missing"
    if not (target / "config.json").exists():
        return False, "config.json is missing"
    if not has_any_file(target, ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")):
        return False, "tokenizer files are missing"
    if not any_file_with_suffix(target, (".safetensors", ".bin", ".pt")):
        return False, "model weight file is missing"
    return True, "qwen files found"


def validate_glm_tokenizer(target: Path, cache_root: Path) -> tuple[bool, str]:
    if not target.exists():
        return False, "target directory is missing"
    if not has_any_file(target, ("config.json", "preprocessor_config.json", "feature_extractor_config.json")):
        return False, "config or feature extractor files are missing"
    if not any_file_with_suffix(target, (".safetensors", ".bin", ".pt")):
        return False, "tokenizer weight file is missing"
    return True, "glm tokenizer files found"


def validate_glm_code(target: Path, cache_root: Path) -> tuple[bool, str]:
    if not target.exists():
        return False, "target directory is missing"
    if not ((target / ".git").exists() or (target / "README.md").exists()):
        return False, ".git or README.md is missing"
    return True, "glm voice code found"


def find_modelscope_model(cache_root: Path, model_name: str) -> Path | None:
    base = cache_root / "modelscope"
    if not base.exists():
        return None
    matches = [path for path in base.rglob(model_name) if path.is_dir()]
    if matches:
        return sorted(matches, key=lambda item: len(str(item)))[0]
    for path in base.rglob("*"):
        if path.is_dir() and path.name == model_name:
            return path
    return None


def validate_paraformer(target: Path, cache_root: Path) -> tuple[bool, str]:
    model_name = "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    found = find_modelscope_model(cache_root, model_name)
    if found is None:
        return False, "ModelScope Paraformer cache directory is missing"
    has_config = any_file_with_suffix(found, (".yaml", ".json"))
    has_weights = any_file_with_suffix(found, (".bin", ".pt", ".pb", ".onnx", ".safetensors"))
    if not has_config:
        return False, f"config files are missing under {found}"
    if not has_weights:
        return False, f"model weight files are missing under {found}"
    return True, f"paraformer files found under {found}"


def validate_soulx_checkpoint(target: Path, cache_root: Path) -> tuple[bool, str]:
    if not target.exists():
        return False, "target directory is missing"
    if not (target / "config.json").exists():
        return False, "config.json is missing"
    if not any_file_with_suffix(target, (".safetensors", ".bin", ".pt")):
        return False, "checkpoint weight file is missing"
    return True, "official checkpoint files found"


ASSETS: dict[str, Asset] = {
    "qwen3-0.6b": Asset(
        key="qwen3-0.6b",
        label="Qwen3-0.6B",
        kind="hf",
        source="Qwen/Qwen3-0.6B",
        target_name="Qwen3-0.6B",
        required=True,
        validator=validate_qwen,
    ),
    "glm-tokenizer": Asset(
        key="glm-tokenizer",
        label="GLM-4-Voice tokenizer",
        kind="hf",
        source="zai-org/glm-4-voice-tokenizer",
        target_name="glm-4-voice-tokenizer",
        required=True,
        validator=validate_glm_tokenizer,
    ),
    "glm-code": Asset(
        key="glm-code",
        label="GLM-4-Voice code",
        kind="git",
        source="https://github.com/zai-org/GLM-4-Voice.git",
        target_name="GLM-4-Voice",
        required=True,
        validator=validate_glm_code,
    ),
    "paraformer": Asset(
        key="paraformer",
        label="Paraformer/FunASR Chinese alignment model",
        kind="modelscope",
        source="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        target_name="modelscope",
        required=True,
        validator=validate_paraformer,
    ),
    "soulx-official": Asset(
        key="soulx-official",
        label="SoulX-Duplug official checkpoint",
        kind="hf",
        source="Soul-AILab/SoulX-Duplug-0.6B",
        target_name="SoulX-Duplug-0.6B",
        required=False,
        validator=validate_soulx_checkpoint,
    ),
}


def target_for(asset: Asset, model_root: Path, cache_root: Path) -> Path:
    if asset.kind == "modelscope":
        return cache_root / "modelscope"
    return model_root / asset.target_name


def remove_partial(asset: Asset, model_root: Path, cache_root: Path) -> None:
    target = target_for(asset, model_root, cache_root)
    if asset.kind == "modelscope":
        found = find_modelscope_model(cache_root, Path(asset.source).name)
        if found and found.exists():
            shutil.rmtree(found)
            log(f"[force] removed {found}")
        return
    if target.exists():
        shutil.rmtree(target)
        log(f"[force] removed {target}")


def normalize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def is_official_hf_endpoint(endpoint: str) -> bool:
    return normalize_endpoint(endpoint) == HF_OFFICIAL_ENDPOINT


def hf_endpoints(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    primary = normalize_endpoint(args.hf_endpoint or env.get("HF_ENDPOINT") or HF_OFFICIAL_ENDPOINT)
    endpoints = [primary]
    fallback = normalize_endpoint(args.hf_fallback_endpoint or "")
    if fallback and fallback not in endpoints:
        endpoints.append(fallback)
    return endpoints


def resolve_hf_token(env: dict[str, str]) -> str | None:
    token = env.get("HF_TOKEN")
    if token:
        return token
    return capture_command(["hf", "auth", "token"], env=os.environ.copy(), cwd=PROJECT_ROOT)


def hf_download(asset: Asset, target: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    if importlib.util.find_spec("huggingface_hub") is not None:
        command_base = [sys.executable, "-m", "huggingface_hub.cli.hf", "download", asset.source, "--local-dir", str(target)]
    elif shutil.which("hf") is not None:
        command_base = ["hf", "download", asset.source, "--local-dir", str(target)]
    else:
        raise RuntimeError(
            "huggingface_hub is not importable and hf CLI is not available. "
            "Activate the conda/base environment and run `python -m pip install -r requirements.txt`."
        )

    token = resolve_hf_token(env)
    failures: list[str] = []
    for endpoint in hf_endpoints(args, env):
        endpoint_env = env.copy()
        endpoint_env["HF_ENDPOINT"] = endpoint
        endpoint_env.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.hf_etag_timeout))
        endpoint_env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hf_download_timeout))

        command = [*command_base, "--max-workers", str(args.hf_max_workers)]
        if args.force:
            command.append("--force-download")
        if token and is_official_hf_endpoint(endpoint):
            endpoint_env["HF_TOKEN"] = token
            endpoint_env.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
        else:
            endpoint_env.pop("HF_TOKEN", None)
            endpoint_env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            endpoint_env["HF_HUB_DISABLE_XET"] = "1"

        log(f"[hf] endpoint={endpoint}")
        if token and not is_official_hf_endpoint(endpoint):
            log("[hf] token not sent to non-official endpoint")

        for attempt in range(1, args.hf_retries + 1):
            try:
                if args.hf_retries > 1:
                    log(f"[hf] attempt {attempt}/{args.hf_retries}")
                run_stream(command, env=endpoint_env, cwd=PROJECT_ROOT)
                return
            except Exception as exc:
                failures.append(f"{endpoint} attempt {attempt}: {exc}")
                if attempt == args.hf_retries:
                    log(f"[warn] hf download failed on {endpoint}: {exc}")
                    break
                sleep_seconds = min(10 * attempt, 30)
                log(f"[warn] hf download failed; retrying in {sleep_seconds}s: {exc}")
                time.sleep(sleep_seconds)

    raise RuntimeError("all Hugging Face download attempts failed:\n" + "\n".join(failures))


def git_download(asset: Asset, target: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is not available")
    run_stream(
        ["git", "clone", "--recurse-submodules", "--progress", asset.source, str(target)],
        env=env,
        cwd=PROJECT_ROOT,
    )


def modelscope_download(asset: Asset, target: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    code = (
        "from modelscope import snapshot_download\n"
        f"path = snapshot_download({asset.source!r}, cache_dir={str(target)!r})\n"
        "print(path)\n"
    )
    run_stream([sys.executable, "-c", code], env=env, cwd=PROJECT_ROOT)


def install_dependencies(env: dict[str, str], *, dry_run: bool) -> None:
    log("[deps] installing runtime dependencies only; install GPU torch with requirements-torch-cu121.txt first if needed")
    command = [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")]
    if dry_run:
        log("[dry-run] " + " ".join(command))
        return
    run_stream(command, env=env, cwd=PROJECT_ROOT)


def selected_assets(args: argparse.Namespace) -> list[Asset]:
    selected: list[str] = []
    if args.all or not args.asset:
        selected.extend(key for key, asset in ASSETS.items() if asset.required)
    selected.extend(args.asset or [])
    if args.include_official_checkpoint:
        selected.append("soulx-official")
    ordered: list[Asset] = []
    seen: set[str] = set()
    for key in selected:
        if key not in ASSETS:
            raise ValueError(f"unknown asset: {key}")
        if key in seen:
            continue
        seen.add(key)
        ordered.append(ASSETS[key])
    return ordered


def verify_python_loads(model_root: Path) -> None:
    try:
        from transformers import AutoFeatureExtractor, AutoTokenizer  # type: ignore
    except ModuleNotFoundError:
        log("[warn] transformers is not installed; skipped lightweight load checks")
        return
    AutoTokenizer.from_pretrained(str(model_root / "Qwen3-0.6B"), trust_remote_code=True)
    AutoFeatureExtractor.from_pretrained(str(model_root / "glm-4-voice-tokenizer"), trust_remote_code=True)
    log("[ok] lightweight transformers load checks passed")


def run_asset(asset: Asset, args: argparse.Namespace, model_root: Path, cache_root: Path, state: dict, env: dict[str, str]) -> bool:
    target = target_for(asset, model_root, cache_root)
    ok, reason = asset.validator(target, cache_root)
    state_entry = state.get("assets", {}).get(asset.key, {})
    if ok:
        log(f"[skip] {asset.label}: {reason}")
        state.setdefault("assets", {})[asset.key] = {
            "source": asset.source,
            "target": str(target),
            "completed_at": state_entry.get("completed_at") or now(),
            "verified_at": now(),
        }
        return True

    if args.verify_only:
        log(f"[missing] {asset.label}: {reason}")
        return False

    if args.dry_run:
        log(f"[start] {asset.label}: {asset.source}")
        log(f"[target] {target}")
        log(f"[dry-run] would download {asset.kind} asset {asset.source}")
        return True

    if target.exists() and asset.kind == "hf" and not args.force:
        log(f"[resume] {asset.label} target exists but is incomplete: {reason}")
        log("[resume] continuing with hf download; existing files will be reused")
    elif target.exists() and asset.kind != "modelscope" and not args.force:
        raise RuntimeError(
            f"{asset.label} target exists but validation failed: {reason}. "
            f"Use --force to remove and re-download: {target}"
        )

    if args.force:
        remove_partial(asset, model_root, cache_root)

    log(f"[start] {asset.label}: {asset.source}")
    log(f"[target] {target}")

    if asset.kind == "hf":
        hf_download(asset, target, args, env)
    elif asset.kind == "git":
        git_download(asset, target, args, env)
    elif asset.kind == "modelscope":
        modelscope_download(asset, target, args, env)
    else:
        raise RuntimeError(f"unsupported asset kind: {asset.kind}")

    ok, reason = asset.validator(target, cache_root)
    if not ok:
        raise RuntimeError(f"{asset.label} download finished but validation failed: {reason}")
    state.setdefault("assets", {})[asset.key] = {
        "source": asset.source,
        "target": str(target),
        "completed_at": now(),
        "verified_at": now(),
    }
    log(f"[done] {asset.label}: {reason}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SoulX-Duplug paper-path model assets.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--all", action="store_true", help="Download required Stage 1/2 assets.")
    parser.add_argument("--asset", action="append", choices=sorted(ASSETS), help="Download or verify one asset. Can be repeated.")
    parser.add_argument("--include-official-checkpoint", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load-check", action="store_true", help="Run lightweight tokenizer/feature-extractor load checks after verification.")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", HF_OFFICIAL_ENDPOINT))
    parser.add_argument("--hf-fallback-endpoint", default="")
    parser.add_argument("--hf-retries", type=int, default=3)
    parser.add_argument("--hf-max-workers", type=int, default=1)
    parser.add_argument("--hf-etag-timeout", type=int, default=int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "60")))
    parser.add_argument("--hf-download-timeout", type=int, default=int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "600")))
    parser.add_argument("--disable-hf-xet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    env = os.environ.copy()
    env["MODEL_ROOT"] = str(model_root)
    env["CACHE_ROOT"] = str(cache_root)
    env["OUTPUT_ROOT"] = str(output_root)
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.hf_etag_timeout))
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hf_download_timeout))
    env.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "2")
    if args.disable_hf_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    else:
        env.pop("HF_HUB_DISABLE_XET", None)

    log(f"MODEL_ROOT={model_root}")
    log(f"CACHE_ROOT={cache_root}")
    log(f"OUTPUT_ROOT={output_root}")
    log(f"HF_HOME={env['HF_HOME']}")
    log(f"MODELSCOPE_CACHE={env['MODELSCOPE_CACHE']}")
    log(f"HF_ENDPOINT={args.hf_endpoint}")
    log(f"HF_FALLBACK_ENDPOINT={args.hf_fallback_endpoint}")
    log(f"HF_HUB_ETAG_TIMEOUT={env['HF_HUB_ETAG_TIMEOUT']}")
    log(f"HF_HUB_DOWNLOAD_TIMEOUT={env['HF_HUB_DOWNLOAD_TIMEOUT']}")
    log(f"HF_XET_NUM_CONCURRENT_RANGE_GETS={env['HF_XET_NUM_CONCURRENT_RANGE_GETS']}")
    if "HF_HUB_DISABLE_XET" in env:
        log(f"HF_HUB_DISABLE_XET={env['HF_HUB_DISABLE_XET']}")

    if not args.dry_run and not args.verify_only:
        model_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["MODELSCOPE_CACHE"]).mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        install_dependencies(env, dry_run=args.dry_run or args.verify_only)

    state = load_state(model_root)
    ok_all = True
    for asset in selected_assets(args):
        try:
            ok_all = run_asset(asset, args, model_root, cache_root, state, env) and ok_all
            if not args.dry_run:
                save_state(model_root, state)
        except KeyboardInterrupt:
            log("\n[interrupt] model download stopped")
            return 130
        except Exception as exc:
            ok_all = False
            log(f"[fail] {asset.label}: {exc}")
            if not args.verify_only:
                return 1

    if args.load_check and ok_all and not args.dry_run:
        try:
            verify_python_loads(model_root)
        except Exception as exc:
            log(f"[fail] lightweight load check failed: {exc}")
            return 1

    if args.verify_only and not ok_all:
        return 1
    log("[complete] model asset task finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
