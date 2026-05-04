#!/usr/bin/env python3
"""
Match narration subtitle lines to the most relevant 10-second movie clips.

By default, subtitles are discovered from /mnt/nas/share/home/lxh/subtitles
using the movie bvid in info_movies.json. The legacy --input-json mode remains
available for explicit per-movie JSON files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

QUERY_TIME = int("1" + "23595999")
MOVIE_GRANULARITIES = ["10sec", "30sec", "3min", "10min"]
DEFAULT_MOVIE_ROOT = Path("/mnt/nas/share/home/lxh")
DEFAULT_SUBTITLE_ROOT = Path("/mnt/nas/share/home/lxh/subtitles")
DEFAULT_WORK_DIR = Path("data/MovieMatch")
DEFAULT_METADATA_DIR = Path("output/metadata/movie_match")
DEFAULT_OUTPUT_DIR = Path("output/movie_match")
DEFAULT_CACHE_DIR = Path(".cache/movie_match")
DEFAULT_MEMORY_MODEL = "gpt-5-mini"
DEFAULT_RETRIEVER_MODEL = "gpt-5-mini"
DEFAULT_RESPOND_MODEL = "gpt-5"
LOCAL_QWEN3VL_MODEL_ALIAS = "qwen3vl-8b"


@dataclass(frozen=True)
class SubtitleLine:
    subtitle_id: int
    text: str


@dataclass(frozen=True)
class SubtitleInput:
    path: Path
    movie_id: int
    language: Optional[str]
    subtitles: list[SubtitleLine]


@dataclass(frozen=True)
class MovieInfo:
    id: int
    title_ch: str
    title_en: str
    source_file: str
    bvid: str
    compressed_file: str


@dataclass(frozen=True)
class MoviePaths:
    movie_path: Path
    transcript_path: Path
    caption_dir: Path
    caption_10sec_path: Path
    metadata_root: Path
    output_path: Path
    cache_root: Path


@dataclass
class ClipCandidate:
    frame_id: str
    start_sec: float
    end_sec: float
    timestamp_sec: float
    caption: str
    sources: set[str] = field(default_factory=set)
    image: Any = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "start_sec": round(self.start_sec, 6),
            "end_sec": round(self.end_sec, 6),
            "timestamp_sec": round(self.timestamp_sec, 6),
            "caption": self.caption,
            "sources": sorted(self.sources),
        }


@dataclass
class RetrievalBundle:
    round_history: list[dict[str, Any]] = field(default_factory=list)
    episodic_entries: list[Any] = field(default_factory=list)
    semantic_entries: list[Any] = field(default_factory=list)
    visual_clips: list[Any] = field(default_factory=list)
    visual_frames: list[Any] = field(default_factory=list)


@dataclass
class SubtitleMatchItem:
    subtitle: SubtitleLine
    bundle: RetrievalBundle
    candidates: list[ClipCandidate]


def format_subtitle_lines(subtitles: list[SubtitleLine]) -> str:
    return "\n".join(f"[#{subtitle.subtitle_id}] {subtitle.text}" for subtitle in subtitles)


def iter_subtitle_batches(
    subtitles: list[SubtitleLine],
    batch_size: int,
    overlap: int,
) -> Iterable[tuple[list[SubtitleLine], list[SubtitleLine]]]:
    batch_size = max(1, batch_size)
    overlap = max(0, overlap)
    total = len(subtitles)

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        context_start = max(0, start - overlap)
        context_end = min(total, end + overlap)
        yield subtitles[start:end], subtitles[context_start:context_end]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def discover_input_jsons(input_paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON path not found: {input_path}")

        if input_path.is_file():
            candidates = [input_path]
        else:
            candidates = sorted(path for path in input_path.rglob("*.json") if path.is_file())

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(candidate)

    if not found:
        raise FileNotFoundError("No input JSON files found.")
    return sorted(found)


def parse_movie_ids(movie_id_args: Optional[list[str]]) -> Optional[list[int]]:
    """Parse optional comma-separated and/or space-separated movie ids."""
    if not movie_id_args:
        return None

    movie_ids: list[int] = []
    for token in movie_id_args:
        for item in token.split(","):
            item = item.strip()
            if item:
                movie_ids.append(int(item))
    return movie_ids or None


def parse_subtitle_input(path: Path, *, fallback_movie_id: Optional[int] = None) -> SubtitleInput:
    data = load_json(path)
    if isinstance(data, list):
        data = {"segments": data}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object or a segment array.")

    if "id" not in data and fallback_movie_id is None:
        raise ValueError(f"{path} is missing required movie id field: id")
    try:
        movie_id = int(data["id"]) if "id" in data else int(fallback_movie_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid movie id: {data.get('id', fallback_movie_id)!r}") from exc

    if fallback_movie_id is not None and movie_id != fallback_movie_id:
        raise ValueError(f"{path} has id={movie_id}, expected id={fallback_movie_id} from info_movies.json.")

    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{path} must contain a non-empty segments array.")

    subtitles: list[SubtitleLine] = []
    for index, segment in enumerate(segments, start=1):
        if isinstance(segment, dict):
            text = segment.get("text")
        elif isinstance(segment, str):
            text = segment
        else:
            raise ValueError(f"{path} segment #{index} must be an object with text.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path} segment #{index} is missing required text.")
        subtitles.append(SubtitleLine(subtitle_id=index, text=text.strip()))

    language = data.get("language")
    return SubtitleInput(
        path=path,
        movie_id=movie_id,
        language=language if isinstance(language, str) else None,
        subtitles=subtitles,
    )


def candidate_subtitle_paths(subtitle_root: Path, bvid: str) -> list[Path]:
    """Return likely subtitle JSON paths for a bvid, exact path first."""
    if not bvid:
        return []
    return [
        subtitle_root / f"{bvid}.json",
        subtitle_root / bvid / f"{bvid}.json",
        subtitle_root / bvid / "subtitle.json",
        subtitle_root / bvid / "subtitles.json",
    ]


def find_subtitle_path_by_bvid(subtitle_root: Path, bvid: str) -> Optional[Path]:
    for candidate in candidate_subtitle_paths(subtitle_root, bvid):
        if candidate.exists() and candidate.is_file():
            return candidate

    if not subtitle_root.exists() or not bvid:
        return None

    matches = sorted(path for path in subtitle_root.rglob("*.json") if path.stem == bvid)
    return matches[0] if matches else None


def discover_subtitle_inputs_from_bvid(
    movie_index: dict[int, MovieInfo],
    subtitle_root: Path,
    *,
    movie_ids: Optional[list[int]] = None,
    dry_run: bool = False,
) -> list[SubtitleInput]:
    selected_movies = [movie_index[movie_id] for movie_id in movie_ids] if movie_ids else list(movie_index.values())
    subtitle_inputs: list[SubtitleInput] = []
    missing_requested: list[str] = []

    for movie in selected_movies:
        if not movie.bvid:
            if movie_ids:
                missing_requested.append(f"id={movie.id}: missing bvid")
            continue

        subtitle_path = find_subtitle_path_by_bvid(subtitle_root, movie.bvid)
        if subtitle_path is None:
            expected = candidate_subtitle_paths(subtitle_root, movie.bvid)[0]
            if dry_run and movie_ids:
                logger.warning("Subtitle JSON not found for movie id=%s bvid=%s; expected %s", movie.id, movie.bvid, expected)
                subtitle_inputs.append(
                    SubtitleInput(
                        path=expected,
                        movie_id=movie.id,
                        language=None,
                        subtitles=[],
                    )
                )
            elif movie_ids:
                missing_requested.append(f"id={movie.id}: {expected}")
            continue

        subtitle_inputs.append(parse_subtitle_input(subtitle_path, fallback_movie_id=movie.id))

    if missing_requested:
        raise FileNotFoundError("Missing subtitle JSON for requested movies:\n" + "\n".join(missing_requested))

    if not subtitle_inputs:
        raise FileNotFoundError(f"No subtitle JSON files found under {subtitle_root} for movies with bvid.")

    return subtitle_inputs


def load_movie_index(movie_info_path: Path) -> dict[int, MovieInfo]:
    data = load_json(movie_info_path)
    movies = data.get("movies") if isinstance(data, dict) else None
    if not isinstance(movies, list):
        raise ValueError(f"{movie_info_path} must contain a movies array.")

    index: dict[int, MovieInfo] = {}
    for raw_movie in movies:
        if not isinstance(raw_movie, dict) or "id" not in raw_movie:
            continue
        movie_id = int(raw_movie["id"])
        if movie_id in index:
            raise ValueError(f"Duplicate movie id in {movie_info_path}: {movie_id}")
        index[movie_id] = MovieInfo(
            id=movie_id,
            title_ch=str(raw_movie.get("title_ch", "")),
            title_en=str(raw_movie.get("title_en", "")),
            source_file=str(raw_movie.get("source_file", "")),
            bvid=str(raw_movie.get("bvid", "")),
            compressed_file=str(raw_movie.get("compressed_file", "")),
        )

    if not index:
        raise ValueError(f"No movie records found in {movie_info_path}.")
    return index


def resolve_movie_path(movie: MovieInfo, movie_root: Path) -> Path:
    raw_path = movie.compressed_file or movie.source_file
    if not raw_path:
        raise ValueError(f"Movie id {movie.id} has no compressed_file or source_file.")

    movie_path = Path(raw_path)
    if movie_path.is_absolute():
        return movie_path
    return movie_root / movie_path


def safe_filename_part(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "movie"


def movie_output_name(movie: MovieInfo) -> str:
    if movie.compressed_file:
        return safe_filename_part(Path(movie.compressed_file).stem)
    if movie.title_en:
        return safe_filename_part(movie.title_en)
    if movie.title_ch:
        return safe_filename_part(movie.title_ch)
    return f"movie_{movie.id}"


def make_movie_paths(args: argparse.Namespace, movie: MovieInfo) -> MoviePaths:
    movie_path = resolve_movie_path(movie, args.movie_root)
    caption_dir = args.work_dir / "caption" / str(movie.id)
    return MoviePaths(
        movie_path=movie_path,
        transcript_path=args.work_dir / "transcript" / f"{movie.id}.srt",
        caption_dir=caption_dir,
        caption_10sec_path=caption_dir / "10sec.json",
        metadata_root=args.metadata_dir,
        output_path=args.output_dir / f"{movie.id}_{movie_output_name(movie)}.json",
        cache_root=args.cache_dir / str(movie.id) / "episodic_memory",
    )


def time_str_to_seconds(time_value: Any) -> float:
    time_str = str(time_value).zfill(8)
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    centiseconds = int(time_str[6:8])
    return float(hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0)


def frame_id_from_start_seconds(start_sec: float) -> str:
    return f"{max(0, int(math.floor(start_sec / 10.0 + 1e-9))):06d}"


def midpoint_seconds(start_sec: float, end_sec: float) -> float:
    return (float(start_sec) + float(end_sec)) / 2.0


def caption_entry_seconds(entry: Any) -> tuple[float, float]:
    return time_str_to_seconds(entry.start_time), time_str_to_seconds(entry.end_time)


def caption_entry_frame_id(entry: Any) -> str:
    start_sec, _ = caption_entry_seconds(entry)
    return frame_id_from_start_seconds(start_sec)


def clip_entry_seconds(clip: Any) -> tuple[float, float]:
    start_sec = clip.clip_start_sec
    end_sec = clip.clip_end_sec
    if start_sec is None:
        start_sec = time_str_to_seconds(clip.start_time)
    if end_sec is None:
        end_sec = time_str_to_seconds(clip.end_time)
    return float(start_sec), float(end_sec)


def ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    logger.info("$ %s", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _expand_optional_path(path: Optional[Path]) -> Optional[Path]:
    return path.expanduser() if path is not None else None


def _require_existing_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _set_path_env(env_name: str, path: Optional[Path], description: str) -> None:
    if path is None:
        return
    _require_existing_path(path, description)
    os.environ[env_name] = str(path)


def configure_local_runtime(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Configure local model paths before spawning builders or loading models."""
    using_local_qwen = args.qwen3vl_model_dir is not None or bool(os.getenv("WORLDMM_QWEN3VL_MODEL"))

    if args.memory_model is None:
        args.memory_model = LOCAL_QWEN3VL_MODEL_ALIAS if using_local_qwen else DEFAULT_MEMORY_MODEL
    if args.retriever_model is None:
        args.retriever_model = LOCAL_QWEN3VL_MODEL_ALIAS if using_local_qwen else DEFAULT_RETRIEVER_MODEL
    if args.respond_model is None:
        args.respond_model = LOCAL_QWEN3VL_MODEL_ALIAS if using_local_qwen else DEFAULT_RESPOND_MODEL

    if args.llm_max_workers is not None and args.llm_max_workers < 1:
        parser.error("--llm-max-workers must be at least 1.")
    if args.caption_workers is not None and args.caption_workers < 1:
        parser.error("--caption-workers must be at least 1.")

    _set_path_env("WORLDMM_QWEN3VL_MODEL", args.qwen3vl_model_dir, "Qwen3-VL model directory")
    _set_path_env("WORLDMM_HF_MODELS_DIR", args.hf_models_dir, "Hugging Face models root")
    _set_path_env("WORLDMM_TEXT_EMBEDDING_MODEL", args.text_embedding_model_dir, "text embedding model directory")
    _set_path_env("WORLDMM_VIS_EMBEDDING_MODEL", args.vis_embedding_model_dir, "visual embedding model directory")
    _set_path_env("WORLDMM_VIS_BASE_MODEL", args.vis_base_model_dir, "visual embedding base model directory")

    if args.whisper_model_dir is not None:
        _require_existing_path(args.whisper_model_dir, "Whisper model directory")
        args.whisper_model = str(args.whisper_model_dir)
        os.environ["WORLDMM_WHISPER_MODEL"] = str(args.whisper_model_dir)

    if args.qwen3vl_device_map:
        os.environ["WORLDMM_QWEN3VL_DEVICE_MAP"] = args.qwen3vl_device_map
    elif using_local_qwen:
        os.environ.setdefault("WORLDMM_QWEN3VL_DEVICE_MAP", "auto")

    if args.llm_max_workers is not None:
        os.environ["WORLDMM_LLM_MAX_WORKERS"] = str(args.llm_max_workers)
    elif using_local_qwen:
        os.environ.setdefault("WORLDMM_LLM_MAX_WORKERS", "1")

    if args.caption_workers is None and using_local_qwen:
        try:
            args.caption_workers = int(os.getenv("WORLDMM_LLM_MAX_WORKERS", "1"))
        except ValueError:
            args.caption_workers = 1

    if args.local_files_only:
        os.environ["WORLDMM_LOCAL_FILES_ONLY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if using_local_qwen:
        logger.info(
            "Local Qwen3-VL runtime: model=%s memory=%s retriever=%s responder=%s device_map=%s llm_workers=%s",
            os.getenv("WORLDMM_QWEN3VL_MODEL", "<hf-cache-or-model-id>"),
            args.memory_model,
            args.retriever_model,
            args.respond_model,
            os.getenv("WORLDMM_QWEN3VL_DEVICE_MAP", "auto"),
            os.getenv("WORLDMM_LLM_MAX_WORKERS", "1"),
        )


def expected_memory_files(paths: MoviePaths, movie_id: int, model_name: str) -> list[Path]:
    return [
        paths.metadata_root / "episodic_memory" / str(movie_id) / f"episodic_triple_results_{model_name}.json",
        paths.metadata_root / "semantic_memory" / str(movie_id) / f"semantic_consolidation_results_{model_name}.json",
        paths.metadata_root / "visual_memory" / str(movie_id) / "visual_embeddings.pkl",
    ]


def ensure_movie_artifacts(args: argparse.Namespace, movie: MovieInfo, paths: MoviePaths) -> None:
    if not paths.movie_path.exists():
        message = f"Movie file not found for id {movie.id}: {paths.movie_path}"
        if args.dry_run:
            logger.warning(message)
        else:
            raise FileNotFoundError(message)

    multiscale_files = [paths.caption_dir / f"{name}.json" for name in ("30sec", "3min", "10min")]
    memory_files = expected_memory_files(paths, movie.id, args.memory_model)
    required_for_matching = [paths.caption_10sec_path, *multiscale_files, *memory_files]

    if args.no_auto_build:
        missing = [path for path in required_for_matching if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(str(path) for path in missing))
        return

    if args.overwrite_artifacts or not paths.transcript_path.exists():
        run_command(
            [
                sys.executable,
                "data/Video-MME/utils/transcribe.py",
                "--input-path",
                str(paths.movie_path),
                "--output-path",
                str(paths.transcript_path),
                "--model",
                args.whisper_model,
                "--batch-size",
                str(args.whisper_batch_size),
            ],
            dry_run=args.dry_run,
        )

    if args.overwrite_artifacts or not paths.caption_10sec_path.exists():
        run_command(
            [
                sys.executable,
                "preprocess/episodic_memory/generate_fine_caption.py",
                "--video-path",
                str(paths.movie_path),
                "--transcript-path",
                str(paths.transcript_path),
                "--output-path",
                str(paths.caption_10sec_path),
                "--model",
                args.memory_model,
                "--unit-time",
                "10",
                *(["--max-workers", str(args.caption_workers)] if args.caption_workers else []),
                *(["--overwrite"] if args.overwrite_artifacts else []),
            ],
            dry_run=args.dry_run,
        )

    if args.overwrite_artifacts or any(not path.exists() for path in multiscale_files):
        run_command(
            [
                sys.executable,
                "-m",
                "worldmm.memory.episodic.multiscale",
                "--caption_dir",
                str(paths.caption_dir),
                "--model",
                args.memory_model,
                "--base_name",
                "10sec.json",
                "--windows",
                "30,180,600",
                "--granularity_names",
                "30sec,3min,10min",
                "--perspective",
                "general",
            ],
            dry_run=args.dry_run,
        )

    if args.overwrite_artifacts or any(not path.exists() for path in memory_files):
        run_command(
            [
                sys.executable,
                "preprocess/build_memory.py",
                "--caption-dir",
                str(args.work_dir / "caption"),
                "--output-dir",
                str(paths.metadata_root),
                "--model",
                args.memory_model,
                "--step",
                "all",
                "--gpu",
                args.gpu,
                "--num-frames",
                str(args.num_frames),
                "--video-ids",
                str(movie.id),
            ],
            dry_run=args.dry_run,
        )


def balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def normalize_clip_match_data(data: Any, candidates: list[ClipCandidate]) -> dict[str, Any]:
    candidate_by_frame = {candidate.frame_id: candidate for candidate in candidates}
    fallback = candidates[0]
    fallback_summary = "未能解析模型输出，使用检索排名最高的候选片段。"

    if not isinstance(data, dict):
        return {
            "frame_id": fallback.frame_id,
            "reason": {"summary": fallback_summary, "timestamp_sec": fallback.timestamp_sec},
        }

    frame_id = str(data.get("frame_id", ""))
    selected = candidate_by_frame.get(frame_id)
    if selected is None:
        selected = fallback
        frame_id = fallback.frame_id

    raw_reason = data.get("reason")
    reason = raw_reason if isinstance(raw_reason, dict) else {}
    summary = reason.get("summary") or data.get("summary") or fallback_summary
    timestamp_sec = reason.get("timestamp_sec", data.get("timestamp_sec", selected.timestamp_sec))
    try:
        timestamp_sec = float(timestamp_sec)
    except (TypeError, ValueError):
        timestamp_sec = selected.timestamp_sec

    if timestamp_sec < selected.start_sec or timestamp_sec > selected.end_sec:
        timestamp_sec = selected.timestamp_sec

    return {
        "frame_id": frame_id,
        "reason": {
            "summary": str(summary).strip(),
            "timestamp_sec": timestamp_sec,
        },
    }


def parse_clip_match_response(response: str, candidates: list[ClipCandidate]) -> dict[str, Any]:
    data: Any = None
    json_text = balanced_json_object(response.strip())
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = None

    return normalize_clip_match_data(data, candidates)


def parse_subtitle_id(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return None


def parse_batch_clip_match_response(response: str, items: list[SubtitleMatchItem]) -> list[dict[str, Any]]:
    data: Any = None
    json_text = balanced_json_object(response.strip())
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = None

    raw_matches = data.get("matches") if isinstance(data, dict) else None
    raw_by_subtitle_id: dict[int, dict[str, Any]] = {}
    if isinstance(raw_matches, list):
        for raw_match in raw_matches:
            if not isinstance(raw_match, dict):
                continue
            subtitle_id = parse_subtitle_id(raw_match.get("subtitle_id"))
            if subtitle_id is not None:
                raw_by_subtitle_id[subtitle_id] = raw_match

    results: list[dict[str, Any]] = []
    for item in items:
        raw_match = raw_by_subtitle_id.get(item.subtitle.subtitle_id)
        match = normalize_clip_match_data(raw_match, item.candidates)
        results.append(
            {
                "subtitle_id": item.subtitle.subtitle_id,
                "text": item.subtitle.text,
                "frame_id": match["frame_id"],
                "reason": match["reason"],
            }
        )
    return results


def format_entry_for_evidence(entry: Any) -> str:
    try:
        start_sec, end_sec = caption_entry_seconds(entry)
        frame_id = caption_entry_frame_id(entry)
        return f"[{entry.granularity} {frame_id} {start_sec:.3f}-{end_sec:.3f}s] {entry.text}"
    except Exception:
        return str(entry)


def format_visual_clip_for_evidence(clip: Any) -> str:
    start_sec, end_sec = clip_entry_seconds(clip)
    return f"[visual {frame_id_from_start_seconds(start_sec)} {start_sec:.3f}-{end_sec:.3f}s] {clip.video_path}"


def read_frame_at(video_path: Path, timestamp_sec: float) -> tuple[Any, float]:
    from decord import VideoReader, cpu
    from PIL import Image

    video_reader = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(video_reader.get_avg_fps() or 0.0)
    if fps <= 0:
        fps = 1.0
    total_frames = len(video_reader)
    frame_index = min(max(int(round(timestamp_sec * fps)), 0), max(total_frames - 1, 0))
    frame = video_reader[frame_index].asnumpy()
    image = Image.fromarray(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image, frame_index / fps


class SubtitleMatcher:
    def __init__(self, args: argparse.Namespace):
        from worldmm.embedding import EmbeddingModel
        from worldmm.llm import LLMModel, PromptTemplateManager
        from worldmm.memory import WorldMemory

        logger.info("Initializing WorldMM models...")
        self.args = args
        self.prompt_template_manager = PromptTemplateManager()
        self.embedding_model = EmbeddingModel()
        self.retriever_llm = LLMModel(model_name=args.retriever_model)
        if args.respond_model.lower() == args.retriever_model.lower():
            self.respond_llm = self.retriever_llm
        else:
            self.respond_llm = LLMModel(model_name=args.respond_model, fps=1)
        self.world_memory = WorldMemory(
            embedding_model=self.embedding_model,
            retriever_llm_model=self.retriever_llm,
            respond_llm_model=self.respond_llm,
            prompt_template_manager=self.prompt_template_manager,
            episodic_granularities=MOVIE_GRANULARITIES,
            episodic_cache_root=str(args.cache_dir / "default" / "episodic_memory"),
            qa_template_name="subtitle_clip_match",
            reasoning_template_name="subtitle_match_reasoning",
            max_rounds=args.max_rounds,
            max_errors=args.max_errors,
        )
        self.world_memory.set_retrieval_top_k(
            episodic=args.episodic_top_k,
            semantic=args.semantic_top_k,
            visual=args.visual_top_k,
        )

    def load_movie_memory(self, movie: MovieInfo, paths: MoviePaths) -> None:
        self.world_memory.reset()
        self.world_memory.episodic_memory.save_dir_root = str(paths.cache_root)

        caption_files = {
            granularity: str(paths.caption_dir / f"{granularity}.json")
            for granularity in MOVIE_GRANULARITIES
            if (paths.caption_dir / f"{granularity}.json").exists()
        }
        if "10sec" not in caption_files:
            raise FileNotFoundError(f"Missing 10sec captions for movie id {movie.id}: {paths.caption_10sec_path}")

        self.world_memory.load_episodic_captions(caption_files=caption_files)

        semantic_file = (
            paths.metadata_root
            / "semantic_memory"
            / str(movie.id)
            / f"semantic_consolidation_results_{self.args.memory_model}.json"
        )
        if semantic_file.exists():
            self.world_memory.load_semantic_triples(file_path=str(semantic_file))
        else:
            logger.warning("Semantic memory not found: %s", semantic_file)

        visual_pkl = paths.metadata_root / "visual_memory" / str(movie.id) / "visual_embeddings.pkl"
        if visual_pkl.exists():
            clips_data = load_json(paths.caption_10sec_path)
            self.world_memory.load_visual_clips(embeddings_path=str(visual_pkl), clips_data=clips_data)
        else:
            logger.warning("Visual memory not found: %s", visual_pkl)

        self.world_memory.index(QUERY_TIME)

    def retrieve_for_subtitle(self, subtitle_text: str) -> RetrievalBundle:
        reasoning_prompt = self.prompt_template_manager.render("subtitle_match_reasoning")
        bundle = RetrievalBundle()
        retrieved_episodic_ids: set[str] = set()
        retrieved_semantic_ids: set[str] = set()
        retrieved_visual_ids: set[str] = set()
        err_count = 0

        for round_num in range(1, self.args.max_rounds + 1):
            history_str = self.world_memory._format_round_history(bundle.round_history)
            user_content = f"""Subtitle:
{subtitle_text}

Round History:
{history_str}

Task:
Step 1: Decide whether to "search" or "answer".
Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."""

            messages = [dict(item) for item in reasoning_prompt]
            messages.append({"role": "user", "content": user_content})

            try:
                response = self.respond_llm.generate(messages)
                reasoning_output = self.world_memory._parse_reasoning_response(response)
            except Exception as exc:
                err_count += 1
                logger.warning("Reasoning failed for subtitle %r: %s", subtitle_text[:40], exc)
                if err_count >= self.args.max_errors:
                    break
                continue

            if reasoning_output.decision == "answer":
                break

            if reasoning_output.decision != "search" or not reasoning_output.selected_memory:
                err_count += 1
                if err_count >= self.args.max_errors:
                    break
                continue

            memory_type = reasoning_output.selected_memory.memory_type
            search_query = reasoning_output.selected_memory.search_query
            content = "[No results]"

            if memory_type == "episodic":
                entries = self._retrieve_episodic(search_query, retrieved_episodic_ids)
                bundle.episodic_entries.extend(entries)
                content = self.world_memory.episodic_memory.retrieve_captions_as_str(entries) if entries else content
            elif memory_type == "semantic":
                entries = self._retrieve_semantic(search_query, retrieved_semantic_ids)
                bundle.semantic_entries.extend(entries)
                content = self.world_memory.semantic_memory.retrieve_triples_as_str(entries) if entries else content
            elif memory_type == "visual":
                clips, frames = self._retrieve_visual(search_query, retrieved_visual_ids)
                bundle.visual_clips.extend(clips)
                bundle.visual_frames.extend(frames)
                if clips:
                    content = "\n".join(format_visual_clip_for_evidence(clip) for clip in clips)
                elif frames:
                    content = f"[{len(frames)} frames retrieved]"
            else:
                err_count += 1
                continue

            bundle.round_history.append(
                {
                    "round_num": round_num,
                    "decision": "search",
                    "memory_type": memory_type,
                    "search_query": search_query,
                    "retrieved_content": content,
                }
            )

        return bundle

    def _retrieve_episodic(self, query: str, retrieved_ids: set[str]) -> list[Any]:
        result = self.world_memory.episodic_memory.retrieve(
            query=query,
            top_k_per_granularity={
                "10sec": max(self.args.episodic_top_k * 2, 8),
                "30sec": max(self.args.episodic_top_k, 5),
                "3min": 5,
                "10min": 3,
            },
            final_top_k=self.args.episodic_top_k,
            as_context=False,
        )
        entries = result if isinstance(result, list) else []
        new_entries = []
        for entry in entries:
            if entry.id not in retrieved_ids:
                retrieved_ids.add(entry.id)
                new_entries.append(entry)
        return new_entries

    def _retrieve_semantic(self, query: str, retrieved_ids: set[str]) -> list[Any]:
        result = self.world_memory.semantic_memory.retrieve(
            query=query,
            top_k=self.args.semantic_top_k,
            as_context=False,
        )
        entries = result if isinstance(result, list) else []
        new_entries = []
        for entry in entries:
            if entry.id not in retrieved_ids:
                retrieved_ids.add(entry.id)
                new_entries.append(entry)
        return new_entries

    def _retrieve_visual(self, query: str, retrieved_ids: set[str]) -> tuple[list[Any], list[Any]]:
        result = self.world_memory.visual_memory.retrieve(
            query=query,
            top_k=self.args.visual_top_k,
            as_context=False,
        )
        items = result if isinstance(result, list) else []
        clips = []
        frames = []
        for item in items:
            if hasattr(item, "clip_start_sec"):
                if item.id not in retrieved_ids:
                    retrieved_ids.add(item.id)
                    clips.append(item)
            else:
                frames.append(item)
        return clips, frames

    def _base_10sec_entries(self) -> list[Any]:
        return self.world_memory.episodic_memory.captions.get("10sec", [])

    def _base_entries_for_range(self, start_sec: float, end_sec: float) -> list[Any]:
        return [
            entry
            for entry in self._base_10sec_entries()
            if ranges_overlap(start_sec, end_sec, *caption_entry_seconds(entry))
        ]

    def _retrieve_base_10sec(self, query: str) -> list[Any]:
        hipporag = self.world_memory.episodic_memory.hipporag.get("10sec")
        if hipporag is None:
            return []

        retrieval_result = hipporag.retrieve(
            queries=[query],
            num_to_retrieve=max(self.args.candidate_top_k * 2, self.args.candidate_top_k),
        )
        if not retrieval_result or not retrieval_result[0].docs:
            return []

        base_entries = self._base_10sec_entries()
        selected: list[Any] = []
        seen_ids: set[str] = set()
        for doc_text in retrieval_result[0].docs:
            for entry in base_entries:
                if entry.text == doc_text and entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    selected.append(entry)
                    break
            if len(selected) >= self.args.candidate_top_k:
                break
        return selected

    def build_candidates(self, subtitle_text: str, bundle: RetrievalBundle) -> list[ClipCandidate]:
        candidates: dict[str, ClipCandidate] = {}

        def add_entry(entry: Any, source: str) -> None:
            start_sec, end_sec = caption_entry_seconds(entry)
            frame_id = frame_id_from_start_seconds(start_sec)
            if frame_id not in candidates:
                candidates[frame_id] = ClipCandidate(
                    frame_id=frame_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    timestamp_sec=midpoint_seconds(start_sec, end_sec),
                    caption=entry.text,
                    sources={source},
                )
            else:
                candidates[frame_id].sources.add(source)

        for entry in bundle.episodic_entries:
            start_sec, end_sec = caption_entry_seconds(entry)
            if entry.granularity == "10sec":
                add_entry(entry, "episodic")
            elif entry.granularity == "30sec":
                for base_entry in self._base_entries_for_range(start_sec, end_sec):
                    add_entry(base_entry, "episodic_30sec")

        for clip in bundle.visual_clips:
            start_sec, end_sec = clip_entry_seconds(clip)
            base_entries = self._base_entries_for_range(start_sec, end_sec)
            for base_entry in base_entries:
                add_entry(base_entry, "visual")

        for base_entry in self._retrieve_base_10sec(subtitle_text):
            add_entry(base_entry, "episodic_10sec_fallback")

        if not candidates:
            base_entries = self._base_10sec_entries()
            if base_entries:
                add_entry(base_entries[0], "fallback_first_clip")

        return list(candidates.values())[: self.args.candidate_top_k]

    def attach_candidate_images(self, movie_path: Path, candidates: list[ClipCandidate]) -> None:
        if self.args.candidate_image_count <= 0:
            return

        for candidate in candidates:
            try:
                image, actual_timestamp = read_frame_at(movie_path, candidate.timestamp_sec)
            except Exception as exc:
                logger.debug("Could not extract candidate image %s: %s", candidate.frame_id, exc)
                continue
            candidate.image = image
            candidate.timestamp_sec = actual_timestamp

    def close_candidate_images(self, candidates: list[ClipCandidate]) -> None:
        for candidate in candidates:
            if candidate.image is not None:
                try:
                    candidate.image.close()
                except Exception:
                    pass
                candidate.image = None

    def evidence_payload_for_bundle(self, bundle: RetrievalBundle) -> dict[str, Any]:
        return {
            "round_history": bundle.round_history,
            "episodic_evidence": [format_entry_for_evidence(entry) for entry in bundle.episodic_entries[:20]],
            "semantic_evidence": [entry.to_display_str() for entry in bundle.semantic_entries[:20]],
            "visual_evidence": [format_visual_clip_for_evidence(clip) for clip in bundle.visual_clips[:20]],
        }

    def prepare_match_item(self, movie_path: Path, subtitle: SubtitleLine) -> SubtitleMatchItem:
        bundle = self.retrieve_for_subtitle(subtitle.text)
        candidates = self.build_candidates(subtitle.text, bundle)
        if not candidates:
            raise RuntimeError(f"No clip candidates found for subtitle #{subtitle.subtitle_id}: {subtitle.text}")

        self.attach_candidate_images(movie_path, candidates)
        return SubtitleMatchItem(subtitle=subtitle, bundle=bundle, candidates=candidates)

    def choose_candidate(self, movie: MovieInfo, movie_path: Path, subtitle: SubtitleLine, candidates: list[ClipCandidate], bundle: RetrievalBundle) -> dict[str, Any]:
        prompt = self.prompt_template_manager.render("subtitle_clip_match")
        candidate_payload = [candidate.to_prompt_dict() for candidate in candidates]
        evidence_payload = self.evidence_payload_for_bundle(bundle)

        text_block = f"""Movie:
id={movie.id}
title_ch={movie.title_ch}
title_en={movie.title_en}
path={movie_path}

Subtitle:
{subtitle.text}

Candidate 10-second clips:
{json.dumps(candidate_payload, indent=2, ensure_ascii=False)}

Retrieved evidence:
{json.dumps(evidence_payload, indent=2, ensure_ascii=False)}

Select exactly one candidate frame_id and return the required JSON object."""

        content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
        for candidate in candidates:
            if candidate.image is None:
                continue
            content.append(
                {
                    "type": "text",
                    "text": f"Visual reference for candidate frame_id={candidate.frame_id}, timestamp_sec={candidate.timestamp_sec:.6f}",
                }
            )
            content.append({"type": "image", "image": candidate.image})

        messages = [dict(item) for item in prompt]
        messages.append({"role": "user", "content": content})
        response = self.respond_llm.generate(messages)
        return parse_clip_match_response(response, candidates)

    def choose_candidate_batch(
        self,
        movie: MovieInfo,
        movie_path: Path,
        target_subtitles: list[SubtitleLine],
        context_subtitles: list[SubtitleLine],
        items: list[SubtitleMatchItem],
    ) -> list[dict[str, Any]]:
        prompt = self.prompt_template_manager.render("subtitle_clip_batch_match")
        subtitle_payload = [
            {
                "subtitle_id": item.subtitle.subtitle_id,
                "text": item.subtitle.text,
                "candidates": [candidate.to_prompt_dict() for candidate in item.candidates],
                "retrieved_evidence": self.evidence_payload_for_bundle(item.bundle),
            }
            for item in items
        ]

        text_block = f"""Movie:
id={movie.id}
title_ch={movie.title_ch}
title_en={movie.title_en}
path={movie_path}

Subtitle context block:
{format_subtitle_lines(context_subtitles)}

Target subtitle_ids:
{json.dumps([subtitle.subtitle_id for subtitle in target_subtitles], ensure_ascii=False)}

Per-target candidates and evidence:
{json.dumps(subtitle_payload, indent=2, ensure_ascii=False)}

Select exactly one candidate frame_id for every target subtitle_id and return the required JSON object."""

        content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
        for item in items:
            for candidate in item.candidates:
                if candidate.image is None:
                    continue
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Visual reference for subtitle_id={item.subtitle.subtitle_id}, "
                            f"frame_id={candidate.frame_id}, timestamp_sec={candidate.timestamp_sec:.6f}"
                        ),
                    }
                )
                content.append({"type": "image", "image": candidate.image})

        messages = [dict(item) for item in prompt]
        messages.append({"role": "user", "content": content})
        response = self.respond_llm.generate(messages)
        return parse_batch_clip_match_response(response, items)

    def match_subtitle(self, movie: MovieInfo, paths: MoviePaths, subtitle: SubtitleLine) -> dict[str, Any]:
        item = self.prepare_match_item(paths.movie_path, subtitle)
        try:
            match = self.choose_candidate(movie, paths.movie_path, subtitle, item.candidates, item.bundle)
        finally:
            self.close_candidate_images(item.candidates)

        return {
            "subtitle_id": subtitle.subtitle_id,
            "text": subtitle.text,
            "frame_id": match["frame_id"],
            "reason": match["reason"],
        }

    def match_subtitle_batch(
        self,
        movie: MovieInfo,
        paths: MoviePaths,
        target_subtitles: list[SubtitleLine],
        context_subtitles: list[SubtitleLine],
    ) -> list[dict[str, Any]]:
        items: list[SubtitleMatchItem] = []
        try:
            for subtitle in target_subtitles:
                items.append(self.prepare_match_item(paths.movie_path, subtitle))
            try:
                return self.choose_candidate_batch(movie, paths.movie_path, target_subtitles, context_subtitles, items)
            except Exception as exc:
                logger.warning(
                    "Batch final matching failed for movie id=%s subtitles %s-%s: %s. Falling back to single matches.",
                    movie.id,
                    target_subtitles[0].subtitle_id if target_subtitles else "?",
                    target_subtitles[-1].subtitle_id if target_subtitles else "?",
                    exc,
                )
                results = []
                for item in items:
                    match = self.choose_candidate(movie, paths.movie_path, item.subtitle, item.candidates, item.bundle)
                    results.append(
                        {
                            "subtitle_id": item.subtitle.subtitle_id,
                            "text": item.subtitle.text,
                            "frame_id": match["frame_id"],
                            "reason": match["reason"],
                        }
                    )
                return results
        finally:
            for item in items:
                self.close_candidate_images(item.candidates)

    def cleanup(self) -> None:
        self.world_memory.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match movie narration subtitles to 10-second WorldMM clips.")
    parser.add_argument("--input-json", nargs="*", type=Path, default=None, help="Legacy mode: one or more subtitle JSON files or directories.")
    parser.add_argument("--subtitle-root", type=Path, default=DEFAULT_SUBTITLE_ROOT, help="Root where subtitle JSON files are discovered by bvid.")
    parser.add_argument("--movie-ids", nargs="*", default=None, help="Optional movie ids to process. Accepts space-separated or comma-separated ids.")
    parser.add_argument("--movie-info", type=Path, default=Path("info_movies.json"), help="Path to info_movies.json.")
    parser.add_argument("--movie-root", type=Path, default=DEFAULT_MOVIE_ROOT, help="Root used for relative compressed_file paths.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Workspace for transcripts and captions.")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR, help="WorldMM metadata root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for per-movie JSON outputs.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="HippoRAG cache root for matching.")
    parser.add_argument("--memory-model", default=None, help="Model used for caption/memory construction. Defaults to gpt-5-mini, or qwen3vl-8b when --qwen3vl-model-dir is set.")
    parser.add_argument("--retriever-model", default=None, help="LLM model for retrieval support. Defaults to gpt-5-mini, or qwen3vl-8b when --qwen3vl-model-dir is set.")
    parser.add_argument("--respond-model", default=None, help="LLM model for reasoning and final matching. Defaults to gpt-5, or qwen3vl-8b when --qwen3vl-model-dir is set.")
    parser.add_argument("--qwen3vl-model-dir", type=Path, default=None, help="Local Qwen3-VL-8B-Instruct directory. Sets WORLDMM_QWEN3VL_MODEL and defaults all LLM roles to qwen3vl-8b.")
    parser.add_argument("--qwen3vl-device-map", default=None, help="Device map for local Qwen3-VL loading, e.g. auto, cuda:0, cuda:1.")
    parser.add_argument("--hf-models-dir", type=Path, default=None, help="Root directory containing local Hugging Face model folders.")
    parser.add_argument("--text-embedding-model-dir", type=Path, default=None, help="Local Qwen3 text embedding model directory.")
    parser.add_argument("--vis-embedding-model-dir", type=Path, default=None, help="Local VLM2Vec adapter directory.")
    parser.add_argument("--vis-base-model-dir", type=Path, default=None, help="Local visual embedding backbone directory, e.g. Qwen2-VL-2B-Instruct.")
    parser.add_argument("--whisper-model-dir", type=Path, default=None, help="Local faster-whisper CTranslate2 model directory.")
    parser.add_argument("--whisper-model", default=os.getenv("WORLDMM_WHISPER_MODEL", "distil-large-v3.5"), help="faster-whisper model for transcript generation.")
    parser.add_argument("--whisper-batch-size", type=int, default=16, help="Concurrent files for transcription.")
    parser.add_argument("--gpu", default="0", help="GPU token list for visual feature extraction.")
    parser.add_argument("--num-frames", type=int, default=10, help="Frames sampled for visual embeddings.")
    parser.add_argument("--llm-max-workers", type=int, default=None, help="Maximum concurrent local LLM calls. Defaults to 1 for local Qwen3-VL.")
    parser.add_argument("--caption-workers", type=int, default=None, help="Maximum concurrent LLM calls during 10-second caption generation. Defaults to --llm-max-workers for local Qwen3-VL.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum WorldMM retrieval rounds per subtitle.")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum retrieval errors before final selection.")
    parser.add_argument("--episodic-top-k", type=int, default=5)
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--visual-top-k", type=int, default=5)
    parser.add_argument("--candidate-top-k", type=int, default=12, help="Maximum candidate 10s clips sent to final prompt.")
    parser.add_argument("--candidate-image-count", type=int, default=1, help="Attach one sampled frame per candidate when > 0.")
    parser.add_argument("--match-batch-size", type=int, default=1, help="Number of target subtitles to send to the final matching prompt at once. Use 30 for contextual batch matching.")
    parser.add_argument("--match-batch-overlap", type=int, default=5, help="Neighboring subtitle count included before and after each final matching batch as context.")
    parser.add_argument("--no-auto-build", action="store_true", help="Do not generate missing WorldMM artifacts.")
    parser.add_argument("--overwrite-artifacts", action="store_true", help="Regenerate transcript, captions, and memory artifacts.")
    parser.add_argument("--local-files-only", action="store_true", help="Force Hugging Face/transformers loaders to use local files only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned actions without loading models.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.match_batch_size < 1:
        parser.error("--match-batch-size must be at least 1.")
    if args.match_batch_overlap < 0:
        parser.error("--match-batch-overlap must be non-negative.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if args.match_batch_size > 1 and args.candidate_image_count > 0:
        logger.warning(
            "Batched final matching may send many images (%s subtitles x up to %s candidates). "
            "Use --candidate-image-count 0 or lower --candidate-top-k if the API rejects large requests.",
            args.match_batch_size,
            args.candidate_top_k,
        )

    args.movie_root = args.movie_root.expanduser()
    args.subtitle_root = args.subtitle_root.expanduser()
    args.work_dir = args.work_dir.expanduser()
    args.metadata_dir = args.metadata_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.cache_dir = args.cache_dir.expanduser()
    args.movie_info = args.movie_info.expanduser()
    args.qwen3vl_model_dir = _expand_optional_path(args.qwen3vl_model_dir)
    args.hf_models_dir = _expand_optional_path(args.hf_models_dir)
    args.text_embedding_model_dir = _expand_optional_path(args.text_embedding_model_dir)
    args.vis_embedding_model_dir = _expand_optional_path(args.vis_embedding_model_dir)
    args.vis_base_model_dir = _expand_optional_path(args.vis_base_model_dir)
    args.whisper_model_dir = _expand_optional_path(args.whisper_model_dir)

    configure_local_runtime(args, parser)

    movie_index = load_movie_index(args.movie_info)
    movie_ids = parse_movie_ids(args.movie_ids)
    if movie_ids:
        missing_movie_ids = [movie_id for movie_id in movie_ids if movie_id not in movie_index]
        if missing_movie_ids:
            raise ValueError(f"Unknown movie ids in {args.movie_info}: {', '.join(str(i) for i in missing_movie_ids)}")

    if args.input_json:
        subtitle_inputs = [parse_subtitle_input(path) for path in discover_input_jsons(args.input_json)]
        if movie_ids:
            requested_set = set(movie_ids)
            subtitle_inputs = [item for item in subtitle_inputs if item.movie_id in requested_set]
            if not subtitle_inputs:
                raise FileNotFoundError("No --input-json subtitles matched the requested --movie-ids.")
    else:
        subtitle_inputs = discover_subtitle_inputs_from_bvid(
            movie_index,
            args.subtitle_root,
            movie_ids=movie_ids,
            dry_run=args.dry_run,
        )
    jobs: list[tuple[SubtitleInput, MovieInfo, MoviePaths]] = []

    for subtitle_input in subtitle_inputs:
        movie = movie_index.get(subtitle_input.movie_id)
        if movie is None:
            raise ValueError(f"{subtitle_input.path} references unknown movie id {subtitle_input.movie_id}.")
        paths = make_movie_paths(args, movie)
        jobs.append((subtitle_input, movie, paths))
        logger.info(
            "Prepared movie id=%s subtitles=%d subtitle=%s movie=%s output=%s",
            movie.id,
            len(subtitle_input.subtitles),
            subtitle_input.path,
            paths.movie_path,
            paths.output_path,
        )

    for _, movie, paths in jobs:
        ensure_movie_artifacts(args, movie, paths)

    if args.dry_run:
        logger.info("Dry run complete. No models were loaded and no files were written.")
        return 0

    matcher = SubtitleMatcher(args)
    try:
        for subtitle_input, movie, paths in jobs:
            logger.info("Loading WorldMM memory for movie id=%s", movie.id)
            matcher.load_movie_memory(movie, paths)

            batches = list(
                iter_subtitle_batches(
                    subtitle_input.subtitles,
                    batch_size=args.match_batch_size,
                    overlap=args.match_batch_overlap,
                )
            )
            try:
                from tqdm import tqdm

                iterator = tqdm(batches, desc=f"Movie {movie.id}", unit="batch")
            except Exception:
                iterator = batches

            results: list[dict[str, Any]] = []
            for target_subtitles, context_subtitles in iterator:
                if args.match_batch_size <= 1:
                    results.extend(matcher.match_subtitle(movie, paths, subtitle) for subtitle in target_subtitles)
                else:
                    results.extend(matcher.match_subtitle_batch(movie, paths, target_subtitles, context_subtitles))
            write_json(paths.output_path, results)
            logger.info("Saved %d matches to %s", len(results), paths.output_path)
    finally:
        matcher.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
