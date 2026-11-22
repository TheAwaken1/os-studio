"""Markup contract for discoverable style and voice controls."""
from pathlib import Path
import unittest
from html.parser import HTMLParser
ROOT = Path(__file__).resolve().parents[1]

class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids[attrs['id']] = (tag, attrs)

class ControlsTests(unittest.TestCase):
    def test_style_library_has_search_filter_and_results(self):
        doc = Elements()
        doc.feed((ROOT / 'web/index.html').read_text(encoding='utf-8'))
        for name in ('btnBrowseStyles', 'styleDlg', 'styleSearch', 'styleCategory', 'styleResults'):
            self.assertIn(name, doc.ids)
        self.assertEqual(doc.ids['styleDlg'][0], 'dialog')

    def test_voice_player_is_separate_and_not_autoplaying(self):
        doc = Elements()
        doc.feed((ROOT / 'web/index.html').read_text(encoding='utf-8'))
        self.assertIn('voicePreview', doc.ids)
        tag, attrs = doc.ids['voiceAudio']
        self.assertEqual(tag, 'audio')
        self.assertIn('controls', attrs)
        self.assertNotIn('autoplay', attrs)

if __name__ == '__main__':
    unittest.main()
