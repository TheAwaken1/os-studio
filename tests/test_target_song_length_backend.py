"""GPU-free target length contracts against BOTH deployed/source copies."""
import ast
import asyncio
import itertools
import re
import sys
import threading
import time
from contextlib import nullcontext
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOTS = [Path(__file__).resolve().parents[1] / 'app']


def extract(root, filename, names, env=None):
    tree = ast.parse((root / filename).read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    namespace = {'re': re, **(env or {})}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(root / filename), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def library(root):
    return extract(root, 'library.py', {'SECTIONS', 'DEFAULT_STRUCTURE', 'DEFAULT_LINES', 'SECONDS_PER_LINE', '_MULTIPLIER', 'clean_structure', 'section_lines', 'lyrics_length_plan', 'lyrics_prompt', 'fit_lyrics', 'lyrics_budget', 'strip_multipliers'})


class LengthPlannerTests(unittest.TestCase):
    def test_auto_preserves_counts_and_custom_is_nearest_balanced(self):
        for root in ROOTS:
            with self.subTest(root=root):
                lib = library(root)
                self.assertTrue(hasattr(lib, 'lyrics_length_plan'), 'missing length planner')
                shape = ['intro', 'verse', 'pre-chorus', 'chorus', 'verse', 'chorus', 'bridge', 'outro']
                for lines in range(2, 9):
                    plan = lib.lyrics_length_plan(shape, lines)
                    self.assertEqual(plan['counts'], [lib.section_lines(s, lines) for s in shape])
                    self.assertIsNone(plan['target_seconds'])
                    self.assertEqual(plan['limit_seconds'], 360)
                plan = lib.lyrics_length_plan(shape, 4, 180)
                counts = plan['counts']
                self.assertEqual(counts[0], 0)
                self.assertEqual(counts[1], counts[4])
                self.assertEqual(counts[3], counts[5])
                self.assertLessEqual(abs(counts[2] * 2 - counts[1]), 2)
                self.assertTrue(all(1 <= c <= 32 for c in counts[1:]))
                self.assertEqual(plan['estimated_seconds'], round(sum(counts) * 3.9 + 8, 1))
                self.assertLessEqual(abs(plan['estimated_seconds'] - 180), 1.95)
                self.assertEqual(plan['total_lines'], sum(counts))
                self.assertEqual(plan['target_seconds'], 180)
                # A repeated-only shape can reach only multiples of occurrence count.
                for target in (30, 60, 120, 359, 360):
                    p = lib.lyrics_length_plan(['chorus'] * 3, 4, target)
                    best = min(abs(3 * c * 3.9 - target) for c in range(1, 33))
                    self.assertAlmostEqual(abs(p['estimated_seconds'] - target), best)
                # Mixed group multiplicities need a globally reachable total,
                # not independently rounded per-section approximations.
                uneven = ['intro'] + ['verse'] * 3 + ['chorus'] * 2
                for target in (30, 31, 60, 119, 180, 359, 360):
                    p = lib.lyrics_length_plan(uneven, 4, target)
                    best = min(abs((3 * a + 2 * b) * 3.9 + 8 - target)
                               for a, b in itertools.product(range(1, 33), repeat=2))
                    self.assertAlmostEqual(abs(p['estimated_seconds'] - target), best)

    def test_custom_counts_control_prompt_cleanup_and_budget(self):
        for root in ROOTS:
            with self.subTest(root=root):
                lib = library(root)
                shape = ['intro', 'verse', 'pre-chorus', 'chorus', 'verse', 'chorus', 'bridge', 'outro']
                counts = lib.lyrics_length_plan(shape, 4, 360)['counts']
                prompt = lib.lyrics_prompt(shape, 4, counts=counts)
                for i, (name, count) in enumerate(zip(shape, counts), 1):
                    self.assertIn(f'{i}. [{name}] - ' + (f'{count} lines' if count else 'no lines, instrumental'), prompt)
                raw = '\n\n'.join('[' + name + ']\n' + '\n'.join(f'{name} line {i}' for i in range(40)) for name in shape)
                fitted = lib.fit_lyrics(raw, shape, 4, counts=counts)
                blocks = fitted.split('\n\n')
                self.assertEqual([b.splitlines()[0] for b in blocks], [f'[{s}]' for s in shape])
                self.assertEqual([len(b.splitlines()) - 1 for b in blocks], counts)
                self.assertEqual(lib.lyrics_budget(shape, 4, counts=counts), max(260, min(3000, 90 + sum(counts) * 24)))
                self.assertGreater(lib.lyrics_budget(shape, 4, counts=counts), 1500)
                self.assertEqual(lib.lyrics_budget(shape, 8), max(260, min(1500, 90 + sum(lib.section_lines(s, 8) for s in shape) * 24)))
                self.assertEqual(lib.fit_lyrics('untagged pasted text', shape, 4, counts=counts), 'untagged pasted text')

    def test_invalid_targets_and_memory_limit(self):
        for root in ROOTS:
            with self.subTest(root=root):
                lib = library(root)
                self.assertTrue(hasattr(lib, 'lyrics_length_plan'), 'missing length planner')
                for bad in (True, False, '60', 60.0, 29, 361, -1):
                    with self.assertRaises(ValueError, msg=repr(bad)):
                        lib.lyrics_length_plan(['verse'], 4, bad)
                with self.assertRaisesRegex(ValueError, '3:00'):
                    lib.lyrics_length_plan(['verse'], 4, 181, max_seconds=180)
                with self.assertRaisesRegex(ValueError, '6:00'):
                    lib.lyrics_length_plan(['verse'], 4, 361)


class AsgiClient:
    """Exercise real routing/validation without optional httpx2 or a server."""
    def __init__(self, app):
        self.app = app

    def post(self, path, json):
        import json as codec
        messages = []
        async def receive():
            return {'type': 'http.request', 'body': codec.dumps(json).encode(), 'more_body': False}
        async def send(message):
            messages.append(message)
        asyncio.run(self.app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                              'method': 'POST', 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                              'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                              'client': ('test', 123), 'server': ('test', 80)}, receive, send))
        body = b''.join(m.get('body', b'') for m in messages).decode()
        return SimpleNamespace(status_code=messages[0]['status'], text=body, json=lambda: codec.loads(body))


class LengthApiTests(unittest.TestCase):
    def api(self, root, gpu=None):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field

        lib = library(root)
        lib.DEFAULT_WRITER = 'fixture'
        lib.LYRICS_WRITERS = [{'id': 'fixture'}]
        start = Mock(return_value={'job': 'test-job'})
        env = dict(library=lib, BaseModel=BaseModel, Field=Field, HTTPException=HTTPException,
                   app=FastAPI(), _gpu=lambda: gpu, _start=start)
        ns = extract(root, 'app.py', {'LyricsBody', 'LyricsPlanBody', '_lyrics_memory', '_lyrics_plan', 'api_lyrics_plan', 'api_lyrics'}, env)
        return ns, AsgiClient(ns.app), start

    def test_read_only_plan_memory_limits_and_strict_input(self):
        for root in ROOTS:
            with self.subTest(root=root):
                for gpu, limit in ((None, 360), ({'total_gb': 13.9}, 180), ({'total_gb': 14}, 360)):
                    ns, client, start = self.api(root, gpu)
                    response = client.post('/api/lyrics-plan', json={'target_seconds': 120})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()['limit_seconds'], limit)
                    self.assertEqual(response.json()['target_seconds'], 120)
                    start.assert_not_called()
                for target in (True, '60', 60.5, 60.0, 29, 361):
                    response = client.post('/api/lyrics-plan', json={'target_seconds': target})
                    self.assertIn(response.status_code, (400, 422), response.text)
                for path in ('/api/lyrics', '/api/lyrics-plan'):
                    for payload in ({'memory': 'invalid'}, {'memory': 'low', 'target_seconds': 181}):
                        ns, client, start = self.api(root)
                        response = client.post(path, json={'idea': 'a song', **payload})
                        self.assertEqual(response.status_code, 400, response.text)
                        if payload['memory'] == 'low':
                            self.assertIn('3:00', response.json()['detail'])
                        start.assert_not_called()

    def test_writer_forwards_plan_inputs_and_old_clients_still_work(self):
        for root in ROOTS:
            with self.subTest(root=root):
                ns, client, start = self.api(root)
                write = Mock(return_value='lyrics')
                start.side_effect = lambda kind, run: run(SimpleNamespace(write_lyrics=write), {})
                response = client.post('/api/lyrics', json={'idea': 'a song', 'target_seconds': 120, 'memory': 'low'})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(write.call_args.kwargs['target_seconds'], 120)
                self.assertEqual(write.call_args.kwargs['memory'], 'low')
                # Resolve unknown Auto consistently with the read-only standard preview.
                client.post('/api/lyrics', json={'idea': 'a song', 'target_seconds': 240})
                self.assertEqual(write.call_args.kwargs['memory'], 'standard')
                def legacy(idea, on_text=None, writer='', structure=None, lines=None):
                    return 'legacy lyrics'
                start.side_effect = lambda kind, run: run(SimpleNamespace(write_lyrics=legacy), {})
                response = client.post('/api/lyrics', json={'idea': 'legacy'})
                self.assertEqual(response.status_code, 200, response.text)


class LengthServiceTests(unittest.TestCase):
    def harness(self, root):
        lib = library(root)
        lib.DEFAULT_WRITER = 'fixture'
        lib.get_writer = lambda name: {'id': 'fixture', 'download_gb': 1}
        raw = '\n\n'.join(f'[{s}]\n' + '\n'.join(f'{s} line {i}' for i in range(40)) for s in ['intro', 'verse', 'pre-chorus', 'chorus', 'verse', 'chorus', 'bridge', 'outro'])
        class Inputs(dict):
            def to(self, device):
                return self
        proc = Mock()
        proc.return_value = Inputs(input_ids=SimpleNamespace(shape=[1, 2]))
        proc.decode.return_value = raw
        proc.apply_chat_template.return_value = 'prompt'
        def generate(**kwargs):
            kwargs['streamer'].on_finalized_text('x' * 120)
            return [[1, 2, 3]]
        model = SimpleNamespace(processor=proc, generate=Mock(side_effect=generate))
        resolve = Mock(side_effect=lambda mode: 'low' if mode in ('auto', 'low') else 'standard')
        begin = Mock()
        env = dict(library=lib, threading=threading, time=time, _run_lock=threading.Lock(),
                   _cancel=threading.Event(), _lyrics_model=model, _lyrics_writer='fixture',
                   resolve_memory=resolve, _begin=begin, _check=lambda: None, tidy_lyrics=lambda text: text,
                   Cancelled=type('Cancelled', (Exception,), {}))
        ns = extract(root, 'service.py', {'write_lyrics', 'Reporter', 'LYRICS_SAMPLING', 'YUE2_MEMORY'}, env)
        return ns, lib, model, proc, resolve, begin

    def test_plan_drives_writer_prompt_cleanup_budget_and_progress(self):
        class Streamer:
            def __init__(self, *args, **kwargs):
                pass
        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        fake_transformers = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list, TextStreamer=Streamer)
        for root in ROOTS:
            with self.subTest(root=root), patch.dict(sys.modules, torch=fake_torch, transformers=fake_transformers):
                ns, lib, model, proc, resolve, begin = self.harness(root)
                shape = ['intro', 'verse', 'pre-chorus', 'chorus', 'verse', 'chorus', 'bridge', 'outro']
                seen = []
                def on_text(text):
                    rep = ns.write_lyrics.__globals__['_reporter']
                    seen.append(rep._step('write')['frac'])
                for target in (120, None):
                    result = ns.write_lyrics('idea', on_text=on_text, structure=shape, lines=4, target_seconds=target, memory='auto')
                    plan = lib.lyrics_length_plan(shape, 4, target, max_seconds=180)
                    counts = plan['counts'] if target is not None else None
                    self.assertEqual(result, lib.fit_lyrics(proc.decode.return_value, shape, 4, counts=counts))
                    self.assertEqual(proc.apply_chat_template.call_args.args[0][0]['content'], lib.lyrics_prompt(shape, 4, counts=counts))
                    self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], lib.lyrics_budget(shape, 4, counts=counts))
                    self.assertAlmostEqual(seen[-1], 120 / max(300, plan['total_lines'] * 48))
                    resolve.assert_called_with('auto')
                model.generate.reset_mock()
                begin.reset_mock()
                for kwargs in ({'target_seconds': 181, 'memory': 'low'}, {'target_seconds': 60, 'memory': 'invalid'}):
                    with self.assertRaises(ValueError):
                        ns.write_lyrics('idea', **kwargs)
                begin.assert_not_called()
                model.generate.assert_not_called()

    def test_resolve_memory_threshold_and_unknown_match_api(self):
        for root in ROOTS:
            for available, gb, expected in ((False, 0, 'standard'), (True, 13.9, 'low'), (True, 14, 'standard')):
                with self.subTest(root=root, gb=gb):
                    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available,
                        get_device_properties=lambda index: SimpleNamespace(total_memory=gb * 2**30)))
                    ns = extract(root, 'service.py', {'YUE2_MEMORY', 'resolve_memory'})
                    with patch.dict(sys.modules, torch=fake_torch):
                        self.assertEqual(ns.resolve_memory('auto'), expected)


if __name__ == '__main__':
    unittest.main()
