from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmm.model_paths import resolve_hf_model_path, resolve_whisper_model_path

MODULE_PATH = ROOT / "eval" / "match_movie_subtitles.py"
SPEC = importlib.util.spec_from_file_location("match_movie_subtitles", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
match_movie_subtitles = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = match_movie_subtitles
SPEC.loader.exec_module(match_movie_subtitles)


class MatchMovieSubtitlesTest(unittest.TestCase):
    def test_parse_subtitle_input_ignores_timestamps_and_assigns_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.json"
            path.write_text(
                json.dumps(
                    {
                        "id": 3,
                        "language": "zh",
                        "segments": [
                            {"start": 0.0, "end": 1.82, "text": "一个大男人在车站碎碎念"},
                            {"timestamp": 5, "text": "不断向人讲述曾经的过往"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            parsed = match_movie_subtitles.parse_subtitle_input(path)

        self.assertEqual(parsed.movie_id, 3)
        self.assertEqual(parsed.language, "zh")
        self.assertEqual([line.subtitle_id for line in parsed.subtitles], [1, 2])
        self.assertEqual(parsed.subtitles[0].text, "一个大男人在车站碎碎念")

    def test_parse_subtitle_input_requires_id_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_id = Path(tmpdir) / "missing_id.json"
            missing_id.write_text(json.dumps({"segments": [{"text": "x"}]}), encoding="utf-8")
            missing_text = Path(tmpdir) / "missing_text.json"
            missing_text.write_text(json.dumps({"id": 3, "segments": [{}]}), encoding="utf-8")

            with self.assertRaises(ValueError):
                match_movie_subtitles.parse_subtitle_input(missing_id)
            with self.assertRaises(ValueError):
                match_movie_subtitles.parse_subtitle_input(missing_text)

    def test_parse_subtitle_input_allows_missing_id_with_fallback_movie_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "BV1xf4y1q7Wp.json"
            path.write_text(
                json.dumps(
                    {
                        "language": "zh",
                        "segments": [{"text": "一个大男人在车站碎碎念"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            parsed = match_movie_subtitles.parse_subtitle_input(path, fallback_movie_id=3)

        self.assertEqual(parsed.movie_id, 3)
        self.assertEqual(parsed.subtitles[0].subtitle_id, 1)
        self.assertEqual(parsed.subtitles[0].text, "一个大男人在车站碎碎念")

    def test_discover_subtitle_inputs_from_bvid_uses_bvid_json(self) -> None:
        movie = match_movie_subtitles.MovieInfo(
            id=3,
            title_ch="阿甘正传",
            title_en="Forrest Gump",
            source_file="/unused/Forrest_Gump.mkv",
            bvid="BV1xf4y1q7Wp",
            compressed_file="movie_zip/Forrest_Gump_360p.mp4",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_root = Path(tmpdir)
            subtitle_path = subtitle_root / "BV1xf4y1q7Wp.json"
            subtitle_path.write_text(
                json.dumps(
                    {
                        "video": "/unused/BV1xf4y1q7Wp.mp4",
                        "segments": [{"text": "一个大男人在车站碎碎念"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            inputs = match_movie_subtitles.discover_subtitle_inputs_from_bvid(
                {3: movie},
                subtitle_root,
                movie_ids=[3],
            )

        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].movie_id, 3)
        self.assertEqual(inputs[0].path.name, "BV1xf4y1q7Wp.json")

    def test_resolve_movie_path_uses_compressed_file_under_movie_root(self) -> None:
        movie = match_movie_subtitles.MovieInfo(
            id=3,
            title_ch="阿甘正传",
            title_en="Forrest Gump",
            source_file="/unused/Forrest_Gump.mkv",
            bvid="BV1xf4y1q7Wp",
            compressed_file="movie_zip/Forrest_Gump_360p.mp4",
        )
        resolved = match_movie_subtitles.resolve_movie_path(movie, Path("/mnt/nas/share/home/lxh"))
        self.assertEqual(resolved, Path("/mnt/nas/share/home/lxh/movie_zip/Forrest_Gump_360p.mp4"))

    def test_load_movie_index_resolves_id_3_forrest_gump_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            info_path = Path(tmpdir) / "info_movies.json"
            info_path.write_text(
                json.dumps(
                    {
                        "movies": [
                            {
                                "id": 3,
                                "title_ch": "阿甘正传",
                                "title_en": "Forrest Gump",
                                "source_file": "/unused/Forrest_Gump.mkv",
                                "bvid": "BV1xf4y1q7Wp",
                                "compressed_file": "movie_zip/Forrest_Gump_360p.mp4",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            movie = match_movie_subtitles.load_movie_index(info_path)[3]
            resolved = match_movie_subtitles.resolve_movie_path(movie, Path("/mnt/nas/share/home/lxh"))

        self.assertEqual(resolved, Path("/mnt/nas/share/home/lxh/movie_zip/Forrest_Gump_360p.mp4"))

    def test_frame_id_and_midpoint_for_230_to_240_second_clip(self) -> None:
        self.assertEqual(match_movie_subtitles.frame_id_from_start_seconds(230.0), "000023")
        self.assertEqual(match_movie_subtitles.midpoint_seconds(230.0, 240.0), 235.0)

    def test_parse_clip_match_response_validates_frame_id_and_timestamp(self) -> None:
        candidates = [
            match_movie_subtitles.ClipCandidate(
                frame_id="000023",
                start_sec=230.0,
                end_sec=240.0,
                timestamp_sec=235.0,
                caption="候车区长椅上的交谈场景",
            )
        ]
        response = json.dumps(
            {
                "frame_id": "000023",
                "reason": {
                    "summary": "男主在候车区和陌生女子交谈。",
                    "timestamp_sec": 234.98475,
                },
            },
            ensure_ascii=False,
        )

        parsed = match_movie_subtitles.parse_clip_match_response(response, candidates)

        self.assertEqual(parsed["frame_id"], "000023")
        self.assertEqual(parsed["reason"]["summary"], "男主在候车区和陌生女子交谈。")
        self.assertEqual(parsed["reason"]["timestamp_sec"], 234.98475)

    def test_parse_clip_match_response_falls_back_to_first_candidate(self) -> None:
        candidates = [
            match_movie_subtitles.ClipCandidate(
                frame_id="000001",
                start_sec=10.0,
                end_sec=20.0,
                timestamp_sec=15.0,
                caption="fallback",
            )
        ]

        parsed = match_movie_subtitles.parse_clip_match_response("not json", candidates)

        self.assertEqual(parsed["frame_id"], "000001")
        self.assertEqual(parsed["reason"]["timestamp_sec"], 15.0)

    def test_resolve_hf_model_path_from_local_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            local_qwen = models_dir / "Qwen"
            local_qwen.mkdir()

            with patch.dict(os.environ, {"WORLDMM_HF_MODELS_DIR": str(models_dir)}):
                resolved = resolve_hf_model_path("Qwen/Qwen3-Embedding-4B")

        self.assertEqual(resolved, str(local_qwen))

    def test_resolve_whisper_model_path_requires_ct2_model_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            original_whisper = models_dir / "distil-whisper"
            original_whisper.mkdir()
            ct2_whisper = models_dir / "distil-large-v3.5-ct2"
            ct2_whisper.mkdir()
            (ct2_whisper / "model.bin").write_text("fake", encoding="utf-8")

            with patch.dict(os.environ, {"WORLDMM_HF_MODELS_DIR": str(models_dir)}):
                resolved = resolve_whisper_model_path("distil-large-v3.5")

        self.assertEqual(resolved, str(ct2_whisper))

    def test_local_qwen3vl_dir_defaults_movie_match_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "Qwen3-VL-8B-Instruct"
            model_dir.mkdir()
            parser = match_movie_subtitles.build_parser()
            args = parser.parse_args(["--qwen3vl-model-dir", str(model_dir)])

            with patch.dict(os.environ, {}, clear=True):
                match_movie_subtitles.configure_local_runtime(args, parser)

                self.assertEqual(args.memory_model, "qwen3vl-8b")
                self.assertEqual(args.retriever_model, "qwen3vl-8b")
                self.assertEqual(args.respond_model, "qwen3vl-8b")
                self.assertEqual(args.caption_workers, 1)
                self.assertEqual(os.environ["WORLDMM_QWEN3VL_MODEL"], str(model_dir))
                self.assertEqual(os.environ["WORLDMM_LLM_MAX_WORKERS"], "1")

    def test_local_qwen3vl_dir_keeps_explicit_movie_match_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "Qwen3-VL-8B-Instruct"
            model_dir.mkdir()
            parser = match_movie_subtitles.build_parser()
            args = parser.parse_args(
                [
                    "--qwen3vl-model-dir",
                    str(model_dir),
                    "--memory-model",
                    "gpt-5-mini",
                    "--retriever-model",
                    "qwen3vl-8B",
                    "--respond-model",
                    "gpt-5",
                    "--llm-max-workers",
                    "2",
                    "--local-files-only",
                ]
            )

            with patch.dict(os.environ, {}, clear=True):
                match_movie_subtitles.configure_local_runtime(args, parser)

                self.assertEqual(args.memory_model, "gpt-5-mini")
                self.assertEqual(args.retriever_model, "qwen3vl-8B")
                self.assertEqual(args.respond_model, "gpt-5")
                self.assertEqual(args.caption_workers, 2)
                self.assertEqual(os.environ["WORLDMM_LLM_MAX_WORKERS"], "2")
                self.assertEqual(os.environ["WORLDMM_LOCAL_FILES_ONLY"], "1")
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
