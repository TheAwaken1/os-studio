import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import check_driver  # noqa: E402


class DriverCheckTests(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(check_driver.driver_major('617.42'), 617)
        self.assertEqual(check_driver.driver_major('580.65.06'), 580)
        self.assertIsNone(check_driver.driver_major('unknown'))

    def test_new_enough_passes(self):
        for version in ('580.65.06', '581.15', '617.42'):
            ok, message = check_driver.check(version)
            self.assertTrue(ok, version)
            self.assertIn(check_driver.PASS, message)

    def test_old_driver_stops_with_reason(self):
        ok, message = check_driver.check('566.36')
        self.assertFalse(ok)
        self.assertTrue(message.startswith(check_driver.FAIL))
        self.assertIn('566.36', message)
        self.assertIn('580', message)
        self.assertIn('nvidia.com', message)
        self.assertNotIn('Error:', message)      # Pinokio's default /Error:/ handler stays out of it

    def test_missing_driver_stops(self):
        ok, message = check_driver.check(None)
        self.assertFalse(ok)
        self.assertIn('no NVIDIA driver', message)

    def test_unreadable_version_does_not_block(self):
        ok, _ = check_driver.check('N/A')
        self.assertTrue(ok)

    def test_markers_match_install_js(self):
        install = (Path(__file__).resolve().parents[1] / 'install.js').read_text(encoding='utf-8')
        self.assertIn(check_driver.FAIL, install)
        self.assertIn(check_driver.PASS, install)
        self.assertLess(install.index('check_driver.py'), install.index('-r requirements.txt'))


if __name__ == '__main__':
    unittest.main()
