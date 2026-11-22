"""Generate machine-local configuration without changing an existing config.

Models: reuse weights another Pinokio app already downloaded (any
<PINOKIO_HOME>/api/*/app/models or */models folder holding YuE2/YuE2-3B),
otherwise use this app's own app/models. Weights are referenced, never copied.

Voices: trained voices are personal, so they always live in this app's own
app/voices folder, never in a (possibly shared) models folder.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
MARKER = Path('YuE2') / 'YuE2-3B' / 'model.safetensors'


def find_shared_models(root=ROOT):
    """Models folder of another Pinokio app that already has the YuE2 song model."""
    root = Path(root).resolve()
    api = root.parent                      # <PINOKIO_HOME>/api
    if not api.is_dir():
        return None
    for app_dir in sorted(api.iterdir()):
        if not app_dir.is_dir() or app_dir == root:
            continue
        for models in (app_dir / 'app' / 'models', app_dir / 'models'):
            try:
                if (models / MARKER).is_file():
                    return models
            except OSError:
                continue
    return None


def configure(root=ROOT, models=None):
    root = Path(root).resolve()
    app = root / 'app'
    target = app / 'config.json'
    if target.exists():
        cfg = json.loads(target.read_text(encoding='utf-8'))
        if 'voices_dir' not in cfg:        # configs written before voices moved out of models
            cfg['voices_dir'] = (app / 'voices').as_posix()
            (app / 'voices').mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
        return cfg
    local = root / 'settings.local.json'
    settings = json.loads(local.read_text(encoding='utf-8')) if local.exists() else {}
    models = Path(models or settings.get('models_dir') or find_shared_models(root) or (app / 'models')).resolve()
    cfg = {
        'app_dir': app.as_posix(), 'models_dir': models.as_posix(),
        'output_dir': (app / 'output/songs').as_posix(),
        'temp_dir': (app / 'output/temp').as_posix(),
        'input_dir': (app / 'input').as_posix(),
        'voices_dir': (app / 'voices').as_posix(),
        'port': 7861, 'window': 'browser',
        'model_roots': {
            'yue2': (models / 'YuE2').as_posix(),
            'sheetsage2': (models / 'SheetSage2').as_posix(),
            'mert2': (models / 'MERT2').as_posix(),
            'text_encoders': (models / 'clip').as_posix(),
            'seed_vc': (models / 'seed_vc').as_posix(),
            'audio_encoders': models.as_posix(),
        },
    }
    for name in ('output/songs', 'output/temp', 'input', 'voices'):
        (app / name).mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    return cfg


def apply_packaging_fixes(root=ROOT):
    """Keep live engine/GPU status from overflowing a narrow Pinokio pane."""
    css = Path(root) / 'app/web/app.css'
    if not css.exists():
        return
    content = css.read_text(encoding='utf-8')
    if 'Pinokio narrow-header fix' not in content:
        content += '\n/* Pinokio narrow-header fix */\n@media (max-width: 720px) {\n  .top { flex-wrap: wrap; padding: 14px 16px; }\n  .top-right { width: 100%; min-width: 0; flex-wrap: wrap; }\n}\n'
        css.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    cfg = configure()
    apply_packaging_fixes()
    print('Models: ' + cfg['models_dir'], flush=True)
    print('Voices: ' + cfg['voices_dir'], flush=True)
    print('Songs: ' + cfg['output_dir'], flush=True)
