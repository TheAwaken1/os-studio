import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import library  # noqa: E402


class InstrumentalStyleTests(unittest.TestCase):
    def test_singer_is_taken_out_and_instrumental_put_in(self):
        cases = {
            'English, pop, upbeat, female vocal, 120 BPM': 'English, instrumental, pop, upbeat, 120 BPM',
            'English, lo-fi hip-hop, mellow piano, male rapper, 82 BPM': 'English, instrumental, lo-fi hip-hop, mellow piano, 82 BPM',
            'English, folk, warm guitar, duet harmonies': 'English, instrumental, folk, warm guitar',
            'English, jazz, tenor sax, raspy female alto': 'English, instrumental, jazz, tenor sax',   # instruments stay
            'English, instrumental, ambient': 'English, instrumental, ambient',                     # no doubling
        }
        for style, expected in cases.items():
            self.assertEqual(library.instrumental_style(style), expected, style)


class SongVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = library.OUTPUT_DIR
        library.OUTPUT_DIR = os.path.abspath(self.tmp.name)
        self.addCleanup(setattr, library, 'OUTPUT_DIR', self.old)
        self.mp3 = os.path.join(library.OUTPUT_DIR, 'Song_00001.mp3')
        for ext in ('.mp3', '.lrc', '.abc', '.ai.mp3', '.instrumental.mp3', '.vocals.mp3'):
            Path(library.sidecar(self.mp3, ext)).write_bytes(b'x')
        Path(library.sidecar(self.mp3, '.json')).write_text(json.dumps({'title': 'Song', 'duration': 3}))

    def test_versions_are_not_listed_as_songs(self):
        songs = library.list_songs()
        self.assertEqual([s['file'] for s in songs], ['Song_00001.mp3'])
        song = songs[0]
        self.assertTrue(song['has_ai'] and song['has_inst'] and song['has_vocals'] and song['has_score'])
        self.assertFalse(song['instrumental'])

    def test_version_path_falls_back_to_the_song(self):
        self.assertTrue(library.version_path(self.mp3, 'inst').endswith('.instrumental.mp3'))
        self.assertTrue(library.version_path(self.mp3, 'vocals').endswith('.vocals.mp3'))
        self.assertEqual(library.version_path(self.mp3, 'nope'), self.mp3)
        os.remove(library.sidecar(self.mp3, '.vocals.mp3'))
        self.assertEqual(library.version_path(self.mp3, 'vocals'), self.mp3)

    def test_delete_removes_every_version(self):
        library.delete_song(library.song_id(self.mp3))
        self.assertEqual(os.listdir(library.OUTPUT_DIR), [])


class NumberedTitleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = library.OUTPUT_DIR
        library.OUTPUT_DIR = os.path.abspath(self.tmp.name)
        self.addCleanup(setattr, library, 'OUTPUT_DIR', self.old)
        self.n = 0

    def add(self, title):
        self.n += 1
        mp3 = os.path.join(library.OUTPUT_DIR, 'f_%05d.mp3' % self.n)
        Path(mp3).write_bytes(b'x')
        Path(library.sidecar(mp3, '.json')).write_text(json.dumps({'title': title}))

    def test_takes_count_up(self):
        title = "I'm A Roomba Not A Horse"
        self.assertEqual(library.numbered_title(title), title)          # first one keeps its name
        self.add(title)
        self.assertEqual(library.numbered_title(title), title + ' 2')
        self.add(title + ' 2')
        self.assertEqual(library.numbered_title(title), title + ' 3')
        self.assertEqual(library.numbered_title("i'm a roomba not a horse"), "i'm a roomba not a horse 3")
        self.assertEqual(library.numbered_title('Roomba'), 'Roomba')     # a different song is untouched

    def test_old_unnumbered_duplicates_count(self):
        for _ in range(4):
            self.add('Monday Coffee')                                      # made before numbering existed
        self.assertEqual(library.numbered_title('Monday Coffee'), 'Monday Coffee 5')

    def test_deleted_take_numbers_are_not_reused_below_the_top(self):
        self.add('Song'); self.add('Song 3')
        self.assertEqual(library.numbered_title('Song'), 'Song 4')


if __name__ == '__main__':
    unittest.main()
