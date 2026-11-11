import json
from pathlib import Path
import unittest

from experiments.av_diagnostics import describe_text, measure


class AdversarialDiagnosticsTests(unittest.TestCase):
    def test_frozen_av_uses_eligible_test_subset(self):
        frozen = json.loads(Path(__file__).with_name("part2_split.json").read_text())
        self.assertEqual(len(frozen["domain_source"]), 477)
        self.assertEqual(len(frozen["av_test"]), 458)
        self.assertEqual(len(frozen["av_folds"]), 935)
        self.assertEqual(set(frozen["av_folds"]), {0, 1, 2})

    def test_format_features_detect_multiline_nonbmp(self):
        plain = describe_text("Київ")
        post = describe_text("Київ\n🗺️ адреса 123")
        self.assertEqual(plain["newlines"], 0)
        self.assertGreater(post["newlines"], 0)
        self.assertGreater(post["nonbmp"], 0)
        self.assertGreater(post["digits"], 0)

    def test_oof_fold_integrity_and_separation(self):
        texts = ["Львів номер " + str(i) for i in range(15)]
        texts += ["Київ\n🗺️ номер " + str(i) for i in range(15)]
        labels = [0] * 15 + [1] * 15
        folds = [i % 3 for i in range(15)] * 2
        result, probabilities = measure(texts, labels, folds)
        self.assertGreater(result["format_only"]["oof_auc"], .8)
        self.assertEqual(len(probabilities["all_measured"]), 30)
        with self.assertRaises(ValueError):
            measure(texts[:-1], labels, folds)


if __name__ == "__main__":
    unittest.main()
