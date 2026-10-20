"""Check the PR labeler's rules (.github/labels/label_prs.py) against the repo."""

import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELER = runpy.run_path(str(REPO_ROOT / ".github/labels/label_prs.py"))
areas_of = LABELER["areas_of"]
labels_for = LABELER["labels_for"]
ALIASES = LABELER["model_aliases"](str(REPO_ROOT), [])


def changed(path, lines):
    return {"filename": path, "additions": lines, "deletions": 0}


class TestPrLabels(unittest.TestCase):
    def test_configs_find_their_model(self):
        for path, label in [
            ("configs/qwen3_5_9b_tp2.yaml", "model/qwen3_5"),
            ("configs/qwen3_8_27b_tp4.yaml", "model/qwen3_5"),  # by registry key
            ("configs/qwen3omni_2gpu.yaml", "model/qwen3_omni"),
            ("configs/qwen3tts_1p7b.yaml", "model/qwen3_tts"),
            ("configs/cosmos3_nano_tp2.yaml", "model/cosmos3"),
        ]:
            self.assertEqual(areas_of(path, ALIASES), (label,), path)

    def test_rust_mirrors_get_both_areas(self):
        self.assertEqual(set(areas_of("rust/src/graph/runtime.rs", ALIASES)), {"area/rust", "area/graph"})
        self.assertEqual(set(areas_of("rust/src/shm.rs", ALIASES)), {"area/rust", "area/transport"})
        self.assertEqual(set(areas_of("rust/server/src/main.rs", ALIASES)), {"area/rust", "area/api"})

    def test_unknown_path_is_other(self):
        self.assertEqual(areas_of("mstar/brand_new_package/x.py", ALIASES), ("area/other",))

    def test_small_layer_touch_is_dropped_but_models_are_not(self):
        labels = labels_for([changed("mstar/engine/resources/kv/manager.py", 5),
                             changed("mstar/model/orpheus/submodules.py", 2),
                             changed("mstar/api_server/app.py", 40)], ALIASES)
        self.assertEqual(labels, {"area/api", "model/orpheus", "size/S"})

    def test_any_change_to_a_core_file_labels_its_area(self):
        labels = labels_for([changed("mstar/worker/worker.py", 6),
                             changed("mstar/engine/fp32_precision.py", 125)], ALIASES)
        self.assertEqual(labels, {"area/worker", "area/engine", "size/M"})

    def test_many_models_collapse(self):
        labels = labels_for([changed(f"mstar/model/{m}/x.py", 1)
                             for m in ("bagel", "orpheus", "qwen3_5", "whisper")], ALIASES)
        self.assertEqual(labels, {"model/many", "size/XS"})

    def test_docs_and_tests_label_only_alone(self):
        self.assertEqual(labels_for([changed("docs/models.rst", 30)], ALIASES),
                         {"area/docs", "size/S"})
        self.assertEqual(labels_for([changed("docs/models.rst", 30), changed("test/modular/test_x.py", 500),
                                     changed("mstar/api_server/app.py", 12)], ALIASES),
                         {"area/api", "size/XS"})


if __name__ == "__main__":
    unittest.main()
