"""Lightweight style library regression checks; no GPU imports."""
import sys
import unittest
from unittest.mock import mock_open, patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library


class StyleLibraryTests(unittest.TestCase):
    def test_source_prompts_preserved_and_grouped(self):
        lines = (Path(library.APP_ROOT) / 'style_prompts.txt').read_text(encoding='utf-8-sig').splitlines()[3:]
        blocks = [b.splitlines() for b in '\n'.join(lines).strip().split('\n\n')]
        presets = library.style_library()
        self.assertEqual([(p['name'], p['style']) for p in presets], [(b[0], ' '.join(b[1:])) for b in blocks])
        self.assertTrue(all(p['category'] for p in presets))
        trailer = next(p for p in presets if p['name'] == 'Movie trailer orchestra')
        self.assertTrue(trailer['instrumental'])
        self.assertFalse(next(p for p in presets if p['name'] == 'Country-trap')['instrumental'])


    def test_quick_pick_duplicates_excluded_but_subgenres_kept(self):
        quick = [{'name': 'Country', 'style': 'basic country'}, {'name': 'Pop', 'style': 'basic pop'}]
        presets = library.style_library(quick)
        names = {p['name'] for p in presets}
        self.assertNotIn('Country (straight)', names)
        self.assertNotIn('Pop (modern radio)', names)
        self.assertTrue({'Country-trap', 'Country-rock', 'Punk rock', 'Hardcore rock'} <= names)
        self.assertEqual(next(p['category'] for p in presets if p['name'] == 'Rave / hard dance'), 'Electronic & Dance')
        self.assertEqual(next(p['category'] for p in presets if p['name'] == 'Punk rock'), 'Rock & Heavy')

    def test_duplicate_names_and_prompts_keep_first_entry(self):
        text = ('STYLE PROMPTS\nNotes\n\nCountry\nEnglish, warm guitar\n\n'
                ' country (straight)\nDifferent country prompt\n\n'
                'Unique\nEnglish, bass\n\nUNIQUE\nOther words\n\n'
                'Another name\n ENGLISH,   BASS\n\n'
                'Quick copy\n English,  pop\n')
        with patch('builtins.open', mock_open(read_data=text)):
            presets = library.style_library([{'name': 'Pop', 'style': 'English, pop'}])
        self.assertEqual([p['name'] for p in presets], ['Country', 'Unique'])
        self.assertEqual(presets[1]['style'], 'English, bass')


if __name__ == '__main__':
    unittest.main()
