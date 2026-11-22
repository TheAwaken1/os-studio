import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import library  # noqa: E402

# A trimmed SheetSage2 score (real output shape: two voices, % sections, chord symbols).
SCORE = '''X:1
T:
M:4/4
L:1/16
Q:1/4=86
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Em
% intro
V: Vocal
"Em"z8"D"z8|"C"z16|
V: Ins
E,2B,2GFEB,D2A,2D2A,D|C,2E,2G,2E,G,3E,2B2AG|
% verse
V: Vocal
"Em"z2BBA2A2"D"A2GB4z|"C"z2B2A2A2A2B2GEE2|"Em"z2B2A2AG"D"AG2B3E2|"C"E2E2G2EA2GG4E2|
V: Ins
Z4|
% instrumental
V: Vocal
Z2|
V: Ins
E2E2GFEB,D2A,2D2F2|E2E2G2EG3E2G2E2|
% chorus
V: Vocal
"Em"A2E2B2A2"D"A2B2A2GA-|"C"A2E2G2EG3BBA2GA-|
V: Ins
Z2|
'''


class ScoreTests(unittest.TestCase):
    def test_summary_reads_key_tempo_and_sung_sections(self):
        s = library.score_summary(SCORE)
        self.assertEqual((s['key'], s['bpm'], s['meter']), ('Em', 86, '4/4'))
        self.assertEqual([x['tag'] for x in s['sections']], ['intro', 'verse', 'interlude', 'chorus'])
        self.assertEqual([x['sung'] for x in s['sections']], [False, True, False, True])
        # 4 sung bars of 4/4 at 86 BPM = 11.2 s -> 3 lines at ~3.9 s a line
        self.assertEqual([x['lines'] for x in s['sections']], [0, 3, 0, 1])

    def test_chords_decide_the_planning_mode(self):
        self.assertTrue(library.score_has_chords(SCORE))
        self.assertFalse(library.score_has_chords(SCORE.replace('"Em"', '').replace('"D"', '').replace('"C"', '')))
        self.assertFalse(library.score_has_chords('V: Vocal clef=treble name="Vocal Melody"\nABc|'))

    def test_section_names_become_yue2_tags(self):
        self.assertEqual(library.cover_tag('silence'), 'intro')
        self.assertEqual(library.cover_tag('Solo'), 'interlude')
        self.assertEqual(library.cover_tag('pre-chorus and chorus'), 'chorus')
        self.assertEqual(library.cover_tag('bridge'), 'bridge')


class LyricsTests(unittest.TestCase):
    def test_words_group_into_lines(self):
        words = [(0.0, 11.5, 'The'), (11.5, 11.8, 'alarm'), (11.8, 12.2, 'screams'), (12.2, 12.5, 'through'),
                 (12.5, 12.8, 'the'), (12.8, 14.0, 'haze'), (14.2, 14.3, 'I'), (14.3, 14.6, 'groan'),
                 (14.6, 15.0, 'but'), (15.0, 15.3, 'sip'), (15.3, 16.8, 'grace,'), (40.0, 40.4, 'Monday'),
                 (40.4, 41.0, 'morning')]
        lines = library.group_words(words)
        self.assertEqual([t for _, t in lines], ['The alarm screams through the haze', 'I groan but sip grace', 'Monday morning'])
        self.assertEqual(lines[0][0], 14.0)          # placed by its end, not the bogus 0.0 start

    def test_lines_land_in_their_sections_by_time(self):
        sections = [{'tag': 'intro', 'start': 0, 'end': 10}, {'tag': 'verse', 'start': 10, 'end': 30},
                    {'tag': 'chorus', 'start': 30, 'end': 60}]
        text = library.cover_lyrics(sections, [(12, 'line one'), (25, 'line two'), (35, 'sing it'), (90, 'tail')])
        self.assertEqual(text, '[intro]\n\n[verse]\nline one\nline two\n\n[chorus]\nsing it\ntail\n')
        self.assertEqual(library.cover_lyrics(sections), '[intro]\n\n[verse]\n\n[chorus]\n')


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = library.COVERS_DIR
        library.COVERS_DIR = os.path.join(self.tmp.name, 'covers')
        self.addCleanup(setattr, library, 'COVERS_DIR', self.old)

    def test_save_list_delete_keeps_outside_files(self):
        os.makedirs(library.COVERS_DIR)
        upload = os.path.join(library.COVERS_DIR, 'x.mp3')
        own_song = os.path.join(self.tmp.name, 'My Song_00001.mp3')       # a cover of one of your songs
        for path in (upload, own_song):
            Path(path).write_bytes(b'x')
        for cid, audio in (('20260101-000000-a', upload), ('20260101-000001-b', own_song)):
            library.save_cover({'id': cid, 'name': cid, 'melody_abc': 'x', 'full_abc': 'y', 'audio': audio, 'created': 1})
        self.assertEqual(len(library.list_covers()), 2)
        self.assertNotIn('melody_abc', library.list_covers()[0])         # scores stay on the server
        library.delete_cover('20260101-000000-a')
        library.delete_cover('20260101-000001-b')
        self.assertFalse(os.path.exists(upload))
        self.assertTrue(os.path.exists(own_song))                        # never deleted
        self.assertEqual(library.list_covers(), [])

    def test_ids_cannot_escape_the_folder(self):
        for bad in ('../secrets', 'a/b', '', 'x' * 200):
            with self.assertRaises(FileNotFoundError):
                library.get_cover(bad)
        self.assertRegex(library.cover_id('Bohemian Rhapsody!'), r'^\d{8}-\d{6}-Bohemian-Rhapsody$')


if __name__ == '__main__':
    unittest.main()
