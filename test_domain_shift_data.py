import unittest

from experiments.domain_shift_data import (
    SEPARATORS, case_manifest, concatenate, eligible_stress_rows, stress_cases,
)


class DomainShiftDataTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"row_id": 10, "text": "Київ біля річки", "spans": [(0, 4)]},
            {"row_id": 11, "text": "Їду до Одеси", "spans": [(7, 12)]},
            {"row_id": 12, "text": "Жодних назв", "spans": []},
            {"row_id": 13, "text": "В центрі Львова", "spans": [(9, 15)]},
            {"row_id": 14, "text": "Тест без місця", "spans": []},
            {"row_id": 15, "text": "Дніпро", "spans": [(0, 6)]},
            {"row_id": 16, "text": "Харків", "spans": [(0, 6)]},
        ]

    def test_preserves_exact_unicode_substrings(self):
        for separator in SEPARATORS:
            merged = concatenate(self.rows[:3], separator)
            self.assertEqual([merged.text[a:b] for a, b in merged.spans], ["Київ", "Одеси"])
            self.assertEqual(merged.source_ids, (10, 11, 12))

    def test_deterministic_and_no_reuse(self):
        left = stress_cases(self.rows)
        right = stress_cases(self.rows)
        self.assertEqual(left, right)
        self.assertEqual(case_manifest(left), case_manifest(right))
        sources = [index for case in left for index in case.source_ids]
        self.assertEqual(len(sources), len(set(sources)))

    def test_invalid_rows_fail_closed(self):
        with self.assertRaises(ValueError):
            concatenate([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            concatenate(self.rows[:2], "INVALID")
        with self.assertRaises(ValueError):
            concatenate([self.rows[0], {"row_id": 90, "text": "Львів", "spans": [(1, 9)]}])

    def test_non_bmp_and_long_offsets(self):
        first = {"row_id": 25, "text": "🗺️ Запоріжжя", "spans": [(3, 12)]}
        second = {"row_id": 26, "text": "Дніпро " * 200, "spans": [(0, 6)]}
        result = concatenate([first, second], "\n\n")
        self.assertEqual([result.text[a:b] for a, b in result.spans], ["Запоріжжя", "Дніпро"])
        self.assertGreater(len(result.text), 128)

    def test_incompatible_source_labels_excluded_only_from_stress(self):
        compatible = [dict(row, bio=[0], issues=[]) for row in self.rows[:3]]
        invalid = dict(self.rows[3], bio=None, issues=["partial_token_boundary"])
        selection, info = eligible_stress_rows(
            compatible + [invalid], fail_on_incompatible=False)
        self.assertEqual(selection, compatible)
        self.assertEqual(info["input_rows"], 4)
        self.assertEqual(info["eligible_rows"], 3)
        self.assertEqual(info["excluded_incompatible_rows"], 1)
        with self.assertRaisesRegex(ValueError, "frozen fit"):
            eligible_stress_rows(compatible + [invalid], fail_on_incompatible=True)


if __name__ == "__main__":
    unittest.main()
