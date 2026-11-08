"""Executable regression checks for non-obvious evaluator behavior."""
import unittest
from metrics import comp_metric, hard_processing, intended_processing, exact_span_f1, alternative_metric


class MetricTests(unittest.TestCase):
    def test_empty_dataset_and_texts(self):
        self.assertEqual(comp_metric([], []), 1)
        self.assertEqual(comp_metric([[]], [[]]), 1)
        self.assertEqual(comp_metric([['Київ']], [[]]), 0)
        self.assertEqual(comp_metric([[]], [['Київ']]), 0)

    def test_duplicates_and_order(self):
        self.assertEqual(comp_metric([['Київ', 'Київ']], [['київ']]), 1)
        self.assertEqual(comp_metric([['Київ', 'Львів']], [['Львів', 'Київ']]), 1)

    def test_addresses_collapse(self):
        self.assertEqual(comp_metric([['вул. Миру 12']], [['вул. Миру 99']]), 1)
        self.assertEqual(alternative_metric([['вул. Миру 12']], [['вул. Миру 99']]), 0)
        self.assertEqual(alternative_metric([[' Київ  ']], [['київ']]), 1)
        with self.assertRaises(ValueError): alternative_metric([[]], [['  ']])

    def test_empty_string_is_an_entity(self):
        self.assertEqual(comp_metric([['12']], [['99']]), 1)
        self.assertEqual(comp_metric([[]], [['12']]), 0)

    def test_literal_regex_is_not_intended_regex(self):
        self.assertEqual(hard_processing('Paris'), '')
        self.assertEqual(comp_metric([['Paris']], [['Cairo']]), 1)
        self.assertEqual(intended_processing('Paris'), 'paris')
        self.assertEqual(hard_processing('Київ!'), 'київ!')
        self.assertEqual(intended_processing('Київ!'), 'київ')

    def test_global_counts(self):
        self.assertEqual(comp_metric([['Київ', 'Львів'], []], [['Київ'], ['Одеса']]), .5)

    def test_document_alignment(self):
        self.assertEqual(comp_metric([['Київ'], ['Львів']], [['Львів'], ['Київ']]), 0)

    def test_boundaries(self):
        self.assertEqual(exact_span_f1([[(0, 4)]], [[(0, 5)]])['f1'], 0)

    def test_length_validation(self):
        with self.assertRaises(ValueError): comp_metric([[]], [])


if __name__ == '__main__':
    unittest.main()
