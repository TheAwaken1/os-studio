import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ConfigurationTests(unittest.TestCase):
    def test_configuration_is_local_and_keeps_user_changes(self):
        spec = importlib.util.spec_from_file_location('configure', ROOT / 'tools/configure.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'app').mkdir()
            models = root / 'shared models'
            models.mkdir()
            cfg = module.configure(root, models)
            self.assertEqual(cfg['app_dir'], (root / 'app').as_posix())
            self.assertEqual(cfg['models_dir'], models.as_posix())
            self.assertEqual(cfg['output_dir'], (root / 'app/output/songs').as_posix())
            self.assertNotIn('engine_venv', cfg)
            cfg['port'] = 7999
            (root / 'app/config.json').write_text(json.dumps(cfg))
            self.assertEqual(module.configure(root, root / 'other')['port'], 7999)
            self.assertEqual(module.configure(root, root / 'other')['models_dir'], models.as_posix())

    def test_packaged_header_fix_is_idempotent(self):
        spec = importlib.util.spec_from_file_location('configure', ROOT / 'tools/configure.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            css = root / 'app/web/app.css'
            css.parent.mkdir(parents=True)
            css.write_text('/* original stylesheet */')
            module.apply_packaging_fixes(root)
            module.apply_packaging_fixes(root)
            content = css.read_text()
            self.assertEqual(content.count('Pinokio narrow-header fix'), 1)
            self.assertIn('flex-wrap: wrap', content)
            self.assertTrue(content.startswith('/* original stylesheet */'))

if __name__ == '__main__':
    unittest.main()
