import unittest
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APPS = [ROOT / 'app']


class PlayerArtworkTests(unittest.TestCase):
    def test_artwork_is_opaque_and_full_bleed_without_dark_padding(self):
        for app in APPS:
            with self.subTest(app=str(app)), Image.open(app / 'web/player-artwork.png') as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.mode, 'RGB')
                for x in (0, 511, 1023):
                    self.assertEqual(image.getpixel((x, 0)), (254, 189, 73))
                    self.assertEqual(image.getpixel((x, 1023)), (154, 221, 216))
                for y in range(1024):
                    for x in (0, 1023):
                        self.assertGreater(min(image.getpixel((x, y))), 50)
                self.assertLess(sum(image.getpixel((512, 360))), 100, 'The music note must remain present.')

    def test_media_metadata_describes_new_artwork_and_busts_old_cache(self):
        for app in APPS:
            with self.subTest(app=str(app)):
                js = (app / 'web/app.js').read_text(encoding='utf-8')
                self.assertIn('/static/player-artwork.png?v=os-studio-1', js)
                self.assertIn('sizes: "1024x1024"', js)


if __name__ == '__main__':
    unittest.main()
