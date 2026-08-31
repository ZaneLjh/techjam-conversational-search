from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import starter.projection as projection_module
from starter.agent import (
    E5_PROJECTION_MANIFEST,
    E5_PROJECTION_SIDECAR,
    Agent,
)
from starter.projection import (
    ProjectionConfig,
    ProjectionIndex,
    canonical_source_sha256,
)
from starter.reranking import RerankingConfig
from tests.test_e4_5_projection import _build_small


class E5PromotionTests(unittest.TestCase):
    def test_source_digest_is_line_ending_invariant(self) -> None:
        canonical_bytes = b"alpha = 1\nbeta = 2\n"
        expected = hashlib.sha256(canonical_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf_source = root / "lf.py"
            crlf_source = root / "crlf.py"
            lf_source.write_bytes(canonical_bytes)
            crlf_source.write_bytes(canonical_bytes.replace(b"\n", b"\r\n"))

            self.assertEqual(canonical_source_sha256(lf_source), expected)
            self.assertEqual(canonical_source_sha256(crlf_source), expected)

    def test_packaged_assets_are_source_relative_and_pinned(self) -> None:
        self.assertTrue(E5_PROJECTION_SIDECAR.is_absolute())
        self.assertTrue(E5_PROJECTION_MANIFEST.is_absolute())
        self.assertEqual(E5_PROJECTION_SIDECAR.parent.name, "assets")
        self.assertEqual(E5_PROJECTION_MANIFEST.parent.name, "assets")
        self.assertEqual(
            hashlib.sha256(E5_PROJECTION_SIDECAR.read_bytes()).hexdigest(),
            "dadffeabfe10e1a4c0dc3f727f0837c7de7015b9b0701c365525df95476edc2a",
        )
        self.assertEqual(
            hashlib.sha256(E5_PROJECTION_MANIFEST.read_bytes()).hexdigest(),
            "65ea2d64383c4d94fde594fdfa3eb47863e5cfb758bb2127b4d27bfa65b3a4d2",
        )
        manifest = json.loads(E5_PROJECTION_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["checksums"]["transform_source_sha256"],
            canonical_source_sha256(Path(projection_module.__file__)),
        )

    def test_default_agent_activates_declared_guarded_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog, sidecar, manifest, _ = _build_small(Path(temporary))
            agent = Agent(
                catalog,
                e5_projection_sidecar=sidecar,
                e5_projection_manifest=manifest,
            )

            self.assertTrue(agent.e5_default_requested)
            self.assertTrue(agent.e5_default_active)
            self.assertEqual(agent.e5_status_reason, "ready")
            self.assertTrue(agent.projection_config.enabled)
            self.assertTrue(agent.projection_index.ready)
            self.assertEqual(agent.projection_config.max_rerank_posterior_size, 1)
            self.assertFalse(agent.projection_config.use_question_rollout)
            self.assertFalse(
                agent.question_policy.config.repeat_other_until_exhausted
            )
            self.assertTrue(agent.reranking_config.enabled)
            self.assertTrue(
                agent.reranking_config.enforce_projection_candidate_membership
            )
            self.assertFalse(agent.reranking_config.use_quality_tiebreak)

    def test_invalid_default_assets_atomically_fall_back_to_frozen_e4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, _, _, _ = _build_small(root)
            promoted = Agent(
                catalog,
                e5_projection_sidecar=root / "missing-sidecar.jsonl.gz",
                e5_projection_manifest=root / "missing-manifest.json",
            )
            fallback = Agent(catalog, enable_e5=False)

            self.assertTrue(promoted.e5_default_requested)
            self.assertFalse(promoted.e5_default_active)
            self.assertTrue(promoted.e5_status_reason.startswith("fallback_to_frozen_e4:"))
            self.assertFalse(promoted.projection_config.enabled)
            self.assertFalse(promoted.reranking_config.enabled)
            self.assertEqual(promoted.projection_index.status_reason, "disabled")

            for agent in (promoted, fallback):
                agent.reset("probe", {})
            message = "I'm looking for Women Shoes. A key requirement is: cotton."
            self.assertEqual(
                promoted.respond("probe", message, 1, 3),
                fallback.respond("probe", message, 1, 3),
            )

    def test_kill_switch_does_not_attempt_projection_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog, sidecar, manifest, _ = _build_small(Path(temporary))
            with patch.object(
                ProjectionIndex,
                "_load",
                side_effect=AssertionError("projection load attempted"),
            ):
                agent = Agent(
                    catalog,
                    enable_e5=False,
                    e5_projection_sidecar=sidecar,
                    e5_projection_manifest=manifest,
                )

            self.assertFalse(agent.e5_default_requested)
            self.assertFalse(agent.e5_default_active)
            self.assertEqual(agent.e5_status_reason, "disabled_by_kill_switch")
            self.assertFalse(agent.projection_config.enabled)
            self.assertFalse(agent.reranking_config.enabled)

    def test_explicit_historical_configuration_is_not_promoted_implicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog, sidecar, manifest, _ = _build_small(Path(temporary))
            agent = Agent(
                catalog,
                projection_config=ProjectionConfig(
                    enabled=True,
                    sidecar_path=str(sidecar),
                    manifest_path=str(manifest),
                    use_question_rollout=True,
                ),
                reranking_config=RerankingConfig(),
            )

            self.assertFalse(agent.e5_default_requested)
            self.assertFalse(agent.e5_default_active)
            self.assertEqual(agent.e5_status_reason, "custom_configuration")
            self.assertTrue(agent.projection_index.ready)
            self.assertTrue(
                agent.question_policy.config.repeat_other_until_exhausted
            )
            self.assertFalse(agent.reranking_config.enabled)


if __name__ == "__main__":
    unittest.main()
