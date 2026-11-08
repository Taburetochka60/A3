import unittest
from analyze_checkpoint_errors import boundary_kind,span_diagnostics,frequency_bin,count_scores
from metrics import comp_metric


class ErrorAnalysisTests(unittest.TestCase):
    def test_ending_and_punctuation(self):
        self.assertEqual(boundary_kind('Києва',(0,5),(0,4)),'word_ending_fragment')
        self.assertEqual(boundary_kind('Київ,',(0,4),(0,5)),'punctuation_or_whitespace')

    def test_neighbor_and_missing_tokens(self):
        text='Біла Церква сьогодні'
        self.assertEqual(boundary_kind(text,(0,11),(0,len(text))),'extra_neighbor_material')
        self.assertEqual(boundary_kind(text,(0,11),(5,11)),'missing_name_material')

    def test_merge_split_and_one_to_one_matching(self):
        text='Київ та Львів'
        merge=span_diagnostics(text,[(0,4),(8,13)],[(0,13)])
        self.assertEqual(len(merge['pairs']),1)
        self.assertEqual(merge['components'][0]['kind'],'merged_locations')
        split=span_diagnostics(text,[(0,13)],[(0,4),(8,13)])
        self.assertEqual(len(split['pairs']),1)
        self.assertEqual(split['components'][0]['kind'],'split_location')

    def test_same_name_wrong_occurrence_hidden_by_surface_metric(self):
        text='Київ і Київ'
        self.assertEqual(comp_metric([['Київ']],[['Київ']]),1)
        diag=span_diagnostics(text,[(0,4)],[(7,11)])
        self.assertEqual(diag['exact_matches'],0)
        self.assertEqual(diag['gold_with_no_pred_overlap'],1)
        self.assertFalse(diag['pairs'])

    def test_frequency_and_micro_counts(self):
        self.assertEqual([frequency_bin(n) for n in [0,1,2,4,5]],['unseen_0','seen_1','seen_2_to_4','seen_2_to_4','seen_5_plus'])
        self.assertAlmostEqual(count_scores(756,182,125)['f1'],.8312259483232546)


if __name__=='__main__':unittest.main()
