import unittest
from russian_corpus_audit import canonical,digest,doc_role,check_row,TOKEN_PATTERN
from ner_data import encode_bio,TOKEN_RE


class RussianCorpusTests(unittest.TestCase):
    def test_same_tokenizer_as_actual_NER(self):
        self.assertEqual(TOKEN_PATTERN,TOKEN_RE.pattern)

    def test_document_split_is_stable_and_independent_of_sentence(self):
        self.assertEqual(doc_role(123456),doc_role(123456))
        roles=[doc_role(i) for i in range(1000)]
        self.assertTrue(50<roles.count('ru_valid')<150)
        self.assertTrue(set(roles)=={'ru_fit','ru_valid'})

    def test_canonical_duplicates_do_not_require_exact_text(self):
        self.assertEqual(digest(canonical(' МОСКВА  ')),digest(canonical('Москва')))
        self.assertNotEqual(digest(' МОСКВА  '),digest('Москва'))
        self.assertNotEqual(canonical('Москва'),canonical('Москве'))

    def test_alignment_matches_production_policy(self):
        samples=[('Москва',[(0,6)]),('Нью-Йорка',[(0,8)]),(' Москва ',[(0,8)]),
                 ('Київ_Львів',[(0,4),(5,10)]),('😀 Москва',[(2,8)]),('Москва',[(0,9)])]
        for text,spans in samples:
            prepared=check_row(text,repr(spans))
            labels,_,_,issues=encode_bio(text,spans)
            self.assertEqual(prepared['eligible'],labels is not None and bool(text.strip()))

    def test_auxiliary_overlap_and_bad_marker_flagging(self):
        row=check_row('МГУ Москвы','[(4,10)]','[(0,10)]','[]')
        self.assertEqual(row['issues']['LOC_ORG_overlaps'],1)
        bad=check_row('Москва','not a literal')
        self.assertFalse(bad['eligible'])
        self.assertEqual(bad['issues']['LOC_parse_failure'],1)


if __name__=='__main__':unittest.main()
