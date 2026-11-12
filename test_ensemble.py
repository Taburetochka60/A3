import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ner_data import Token
from pilot_ner import documents
from part2 import score
from metrics import score_details
from ensemble import blend, bootstrap_delta, document_counts, greedy_search, run_search, subset_search, sweep
from ensemble_data import compose_frame, compose_posts
from ensemble_members import load_bundle, save_bundle, to_reference

TEXTS = [('Вибух у Києві та Львові.', [(8, 13), (17, 23)]),
         ('Новини з Харкова', [(9, 16)]),
         ('Без локацій тут', [])]


def docs():
    frame = pd.DataFrame(dict(text=[t for t, _ in TEXTS], loc_markers=[json.dumps(s) for _, s in TEXTS]),
                         index=[10, 11, 12])
    return documents(frame)


def oracle(items, confidence=.95):
    result = []
    for d in items:
        p = np.full((len(d['tokens']), 3), (1-confidence)/2)
        p[np.arange(len(d['tokens'])), d['bio']] = confidence
        result.append(p)
    return result


def part2_locations(items, p):
    from part2 import decode
    return decode(items, p, 0., 'strict')[0]


def noise(items, seed):
    rng = np.random.default_rng(seed)
    return [rng.dirichlet(np.ones(3), len(d['tokens'])) for d in items]


class ReferenceAlignmentTest(unittest.TestCase):
    def test_first_piece_of_each_word(self):
        reference = [Token('Києві', 0, 5), Token('.', 5, 6)]
        pieces = [Token('Киє', 0, 3), Token('ві', 3, 5), Token('.', 5, 6)]
        p = np.array([[.1, .8, .1], [.1, .1, .8], [.9, .05, .05]])
        np.testing.assert_allclose(to_reference(reference, pieces, p), p[[0, 2]])

    def test_rejects_pieces_crossing_words(self):
        reference = [Token('Київ', 0, 4), Token('і', 5, 6)]
        with self.assertRaises(ValueError):
            to_reference(reference, [Token('Київ і', 0, 6)], np.ones((1, 3))/3)


class MultilinePostTest(unittest.TestCase):
    def test_posts_reach_target_lengths(self):
        posts = compose_posts([5, 6, 7, 8], [3, 3, 3, 3], [6], np.random.default_rng(0))
        self.assertEqual([p['rows'] for p in posts], [[5, 6], [7, 8]])

    def test_spans_point_to_same_surfaces(self):
        rows = pd.DataFrame(dict(text=[t for t, _ in TEXTS], loc_markers=[json.dumps(s) for _, s in TEXTS]),
                            index=[10, 11, 12])
        frame = compose_frame(rows, [dict(rows=[10, 11, 12], separator='\n\n')])
        (doc,) = documents(frame)
        self.assertEqual(doc['row_id'], 10)
        self.assertEqual(doc['text'].count('\n\n'), 2)
        self.assertEqual([doc['text'][a:b] for a, b in doc['spans']], ['Києві', 'Львові', 'Харкова'])


class GroupingHashTest(unittest.TestCase):
    def test_pinned_hash_matches_frozen_split(self):
        import part2_data
        from part2_compat import pin_grouping_hash
        original = part2_data.grouping_hash
        try:
            pin_grouping_hash()
            self.assertEqual(part2_data.read_split('part2_split.json')['grouping']['implementation_sha256'],
                             part2_data.grouping_hash())
        finally:
            part2_data.grouping_hash = original


class BundleTest(unittest.TestCase):
    def test_round_trip_and_mismatch(self):
        items = docs(); p = oracle(items)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'m'/'checkpoint.npz'
            save_bundle(path, items, p, dict(member='m'))
            loaded, meta = load_bundle(path, items)
            self.assertEqual(meta['member'], 'm')
            for a, b in zip(loaded, p):
                np.testing.assert_allclose(a, b, atol=1e-6)
            with self.assertRaises(ValueError):
                load_bundle(path, items[::-1])


class ExportTest(unittest.TestCase):
    def test_part2_checkpoint_to_bundles(self):
        import torch
        from ensemble_members import export_member
        from recurrent_model import NERModel
        from part2 import CONFIG
        items = docs()
        encoding = dict(kind='word', vocabulary={'києві': 2, 'у': 3}, size=4)
        torch.manual_seed(0)
        model = NERModel(encoding['size'], **CONFIG['encoder'])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'best.pt'
            torch.save(dict(model=model.state_dict(), config=CONFIG, encoding=encoding, epoch=1, split_hash='x'), checkpoint)
            member = dict(name='word', kind='part2', checkpoint=checkpoint)
            export_member(member, dict(checkpoint=items, test=items), Path(folder)/'bundles')
            loaded, meta = load_bundle(Path(folder)/'bundles'/'word'/'test.npz', items)
            self.assertEqual(meta['model']['tokenizer'], 'word')
            self.assertEqual([len(p) for p in loaded], [len(d['tokens']) for d in items])
            torch.save(dict(model=model.state_dict(), config=CONFIG, encoding=encoding, epoch=2, split_hash='x'), checkpoint)
            with self.assertRaises(ValueError):
                export_member(member, dict(checkpoint=items), Path(folder)/'bundles')

    def test_bpe_member_is_verified_on_its_own_tokens(self):
        import torch
        from ensemble_members import export_member, part2_predictor
        from recurrent_model import NERModel
        from part2 import CONFIG
        from part2_data import fit_encoding
        items = docs()
        encoding = fit_encoding(items, 'bpe')
        torch.manual_seed(0)
        model = NERModel(encoding['size'], **CONFIG['encoder'])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'best.pt'
            torch.save(dict(model=model.state_dict(), config=CONFIG, encoding=encoding, epoch=1, split_hash='x'), checkpoint)
            encoded, probabilities = part2_predictor(checkpoint)[0](items, native=True)
            self.assertGreater(sum(map(len, encoded)), 0)
            logged = score(encoded, probabilities)['f1']         # what part2.train logs for a BPE member
            member = dict(name='bpe', kind='part2', checkpoint=checkpoint,
                          expected=dict(split='checkpoint', threshold=0., policy='original', f1=logged, tokens='member'))
            self.assertAlmostEqual(export_member(member, dict(checkpoint=items), Path(folder)/'bundles'), logged)
            loaded, _ = load_bundle(Path(folder)/'bundles'/'bpe'/'checkpoint.npz', items)
            self.assertEqual([len(p) for p in loaded], [len(d['tokens']) for d in items])


class TrainMemberTest(unittest.TestCase):
    def test_validation_metrics_no_longer_clash_with_precision_mode(self):
        import part2
        from ensemble_members import train_member
        items = docs(); original = part2.train
        part2.train = lambda: dict(precision='bf16', **part2.score(items, oracle(items)))
        try:
            record = train_member()
        finally:
            part2.train = original
        self.assertEqual(record['precision'], 'bf16')
        self.assertEqual(record['span_precision'], 1.)
        self.assertEqual(part2.score.__module__, 'part2')

    def test_wandb_failure_does_not_stop_training(self):
        import part2
        from ensemble_members import train_member
        original = part2.train

        def train():
            if part2.start_run(None, 'abcdefgh', dict(name='x'), {}) is not None:
                raise AssertionError('expected no run')
            return 'trained'
        def refuse(*a, **k):
            raise RuntimeError('run ID abcdefgh is in use')
        start, part2.start_run, part2.train = part2.start_run, refuse, train
        try:
            with self.assertWarns(UserWarning):
                self.assertEqual(train_member(), 'trained')
            self.assertIs(part2.start_run, refuse)
        finally:
            part2.start_run, part2.train = start, original


class SearchTest(unittest.TestCase):
    def test_sweep_matches_part2_score(self):
        items = docs(); p = blend(dict(a=oracle(items, .7), b=noise(items, 0)), dict(a=2., b=1.))
        for row in sweep(items, p, [0., .5, .8]):
            self.assertEqual(row['f1'], score(items, p, row['threshold'], 'strict')['f1'])

    def test_document_counts_sum_to_metric(self):
        items = docs(); p = noise(items, 5)
        counts = document_counts(items, p, 0.)
        total = score_details([[d['text'][a:b] for a, b in d['spans']] for d in items],
                              part2_locations(items, p))
        self.assertEqual(counts.sum(0).tolist(), [total['tp'], total['fp'], total['fn']])

    def test_bootstrap_of_identical_members_is_zero(self):
        items = docs(); counts = document_counts(items, oracle(items), .5)
        r = bootstrap_delta(counts, counts, samples=100)
        self.assertEqual((r['delta'], r['low'], r['high']), (0., 0., 0.))

    def test_blend_is_normalised_average(self):
        items = docs(); a, b = oracle(items), noise(items, 1)
        mixed = blend(dict(a=a, b=b, c=b), dict(a=1., b=3., c=0.))
        np.testing.assert_allclose(mixed[0], (a[0]+3*b[0])/4)

    def test_search_prefers_accurate_member(self):
        items = docs(); members = dict(good=oracle(items), bad1=noise(items, 2), bad2=noise(items, 3))
        trace = greedy_search(members, items, rounds=3)
        self.assertEqual(trace.added.iloc[0], 'good')
        self.assertEqual(subset_search(members, items).iloc[0].members, 'good')

    def test_run_search_writes_recipe(self):
        items = docs(); members = dict(good=oracle(items), bad=noise(items, 4))
        with tempfile.TemporaryDirectory() as folder:
            for name, p in members.items():
                for split in ['testlike_select', 'checkpoint']:
                    save_bundle(Path(folder)/'bundles'/name/f'{split}.npz', items, p, dict(member=name))
            result, table = run_search(Path(folder)/'bundles', list(members),
                                       dict(testlike_select=items, checkpoint=items), Path(folder)/'search', rounds=2)
            self.assertEqual(list(result['final']['weights']), ['good'])
            self.assertEqual(result['final']['select_f1'], 1.)
            self.assertTrue((Path(folder)/'search'/'recipe.json').exists())
            self.assertEqual(set(table.split), {'testlike_select', 'checkpoint'})


class PostprocessTest(unittest.TestCase):
    POST = 'ДТП біля ТЦ Аркадія\n\n📍 вул. Солом\'янська, 28\nПиши @kyiv_news\nКиїв — пряма трансляція 👈'

    def spans(self, *values):
        return [(self.POST.index(v), self.POST.index(v)+len(v)) for v in values]

    def test_spans_stop_at_line_breaks_and_lose_trailing_junk(self):
        from postprocess import clean_span
        (a, b), = self.spans('ТЦ Аркадія\n\n📍')
        self.assertEqual(self.POST[slice(*clean_span(self.POST, a, b))], 'ТЦ Аркадія')
        (a, b), = self.spans("вул. Солом'янська, 28\nПиши")
        self.assertEqual(self.POST[slice(*clean_span(self.POST, a, b))], "вул. Солом'янська, 28")
        self.assertIsNone(clean_span('📍\nКиїв', 0, 1))

    def test_quotes_hyphenated_names_and_numbers(self):
        from postprocess import clean_span
        def clean(text, value):
            a = text.index(value); span = clean_span(text, a, a+len(value))
            return span and text[slice(*span)]
        self.assertEqual(clean('у парку «Центральний» 5', 'парку «Центральний»'), 'парку «Центральний»')
        self.assertEqual(clean('з Донецьк" і', 'Донецьк"'), 'Донецьк')
        self.assertEqual(clean('в Івано-Франківську.', '-Франківську'), 'Івано-Франківську')
        self.assertEqual(clean('Києво-Святошинському районі', 'Києво-'), 'Києво-Святошинському')
        self.assertEqual(clean("вул. Мар‘яненка 85", '‘яненка 85'), "Мар‘яненка 85")
        self.assertIsNone(clean('школа № 38-', '№ 38-'))
        self.assertEqual(clean('Маршрут 38-ТР', 'ТР'), 'ТР')
        self.assertEqual(clean('у Пуща-Водиці', 'Водиці'), 'Пуща-Водиці')
        self.assertIsNone(clean('Маршрут № 38-ТР', '№ 38-'))
        self.assertEqual(clean('фон:\n\nЧорнобиль - 19\nКиїв - 12', '\n\nЧорнобиль - 19\n'), 'Чорнобиль')
        self.assertEqual(clean('Наші жителі\n\nХуйовий Київ', '\n\nХуйовий Київ'), 'Хуйовий Київ')

    def test_signature_lines_handles_and_links_are_dropped(self):
        from postprocess import postprocess, signature_lines
        texts = [self.POST]+[f'Новина {i}\nКиїв — пряма трансляція 👈' for i in range(3)]
        signatures = signature_lines(texts)
        self.assertEqual(signatures, {'Київ — пряма трансляція 👈'})
        spans = self.spans('ТЦ Аркадія', 'kyiv', 'Київ —')
        self.assertEqual([self.POST[a:b] for a, b in postprocess(self.POST, spans, signatures)], ['ТЦ Аркадія'])
        self.assertEqual(len(postprocess(self.POST, spans, signatures, rules='A')), 3)
        self.assertEqual(postprocess('див. https://kyiv.ua/news', [(13, 17)], set()), [])

    def test_submission_is_rewritten_from_its_spans_file(self):
        from postprocess import postprocess_submission
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            pd.DataFrame(dict(text_id=[0, 1], text=[self.POST, 'У Києві тихо'], locations=['[]', '[]'])).to_csv(folder/'test.csv', index=False)
            pd.DataFrame(dict(text_id=[0, 1], spans=[json.dumps([list(s) for s in self.spans('ТЦ Аркадія\n\n📍', 'kyiv')]), '[[2, 7]]'],
                              confidence=['[1, 1]', '[1]'])).to_csv(folder/'sub_spans.csv', index=False)
            stats = postprocess_submission(folder/'sub_spans.csv', folder/'test.csv', folder/'out.csv')
            out = pd.read_csv(folder/'out.csv')
            self.assertEqual([json.loads(x) for x in out.locations], [['ТЦ Аркадія'], ['Києві']])
            self.assertEqual((stats['spans_in'], stats['spans_out'], stats['changed']), (3, 2, 2))
            pd.DataFrame(dict(text_id=[1, 0], spans=['[]', '[]'])).to_csv(folder/'bad_spans.csv', index=False)
            with self.assertRaises(ValueError):
                postprocess_submission(folder/'bad_spans.csv', folder/'test.csv', folder/'bad.csv')


if __name__ == '__main__':
    unittest.main()
