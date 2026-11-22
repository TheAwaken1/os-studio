import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import update_check  # noqa: E402

OLD = 'a' * 40
NEW = 'b' * 40


def fake_repo(root, head='ref: refs/heads/main\n', loose=OLD, packed=None,
              url='https://github.com/someone/os-studio.git'):
    git = os.path.join(root, '.git')
    os.makedirs(os.path.join(git, 'refs', 'heads'))
    Path(git, 'HEAD').write_text(head)
    if loose:
        Path(git, 'refs', 'heads', 'main').write_text(loose + '\n')
    if packed:
        Path(git, 'packed-refs').write_text('# pack-refs with: peeled\n%s refs/heads/main\n' % packed)
    Path(git, 'config').write_text('[core]\n\tbare = false\n[remote "origin"]\n\turl = %s\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n' % url)
    return root


class UpdateCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []

    def fetch_with(self, compare, release=None):
        def fetch(url):
            self.calls.append(url)
            if '/compare/' in url:
                if isinstance(compare, Exception):
                    raise compare
                return compare
            if release is None:
                raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)
            return release
        return fetch

    def test_reads_loose_and_packed_refs_and_remote(self):
        git = os.path.join(fake_repo(self.tmp.name, loose=None, packed=OLD), '.git')
        self.assertEqual(update_check.read_head(git), (OLD, 'main'))
        self.assertEqual(update_check.github_repo(git), 'someone/os-studio')

    def test_ssh_and_non_github_remotes(self):
        root = fake_repo(self.tmp.name, url='git@github.com:someone/os-studio.git')
        self.assertEqual(update_check.github_repo(os.path.join(root, '.git')), 'someone/os-studio')
        other = tempfile.mkdtemp(dir=self.tmp.name)
        fake_repo(other, url='https://gitlab.com/someone/os-studio.git')
        self.assertFalse(update_check.check(other, self.fetch_with({}))['available'])
        self.assertEqual(self.calls, [])                     # nothing is sent anywhere

    def test_newer_commits_with_release(self):
        root = fake_repo(self.tmp.name)
        info = update_check.check(root, self.fetch_with(
            {'status': 'ahead', 'ahead_by': 2, 'commits': [{'sha': 'c' * 40}, {'sha': NEW}]},
            {'tag_name': 'v1.2', 'name': 'v1.2: Takes', 'html_url': 'https://github.com/someone/os-studio/releases/tag/v1.2'}))
        self.assertTrue(info['available'])
        self.assertEqual((info['behind'], info['latest'], info['current']), (2, NEW[:7], OLD[:7]))
        self.assertEqual(info['release']['name'], 'v1.2: Takes')
        self.assertIn('/compare/%s...main' % OLD, self.calls[0])

    def test_newer_commits_without_release_links_the_compare_page(self):
        info = update_check.check(fake_repo(self.tmp.name), self.fetch_with(
            {'status': 'ahead', 'ahead_by': 1, 'commits': [{'sha': NEW}]}))
        self.assertTrue(info['available'])
        self.assertIsNone(info['release'])
        self.assertEqual(info['url'], 'https://github.com/someone/os-studio/compare/%s...main' % OLD[:7])

    def test_up_to_date_offline_or_unknown_commit_stay_quiet(self):
        root = fake_repo(self.tmp.name)
        self.assertFalse(update_check.check(root, self.fetch_with({'status': 'identical', 'ahead_by': 0}))['available'])
        self.assertFalse(update_check.check(root, self.fetch_with({'status': 'behind', 'ahead_by': 0}))['available'])
        self.assertFalse(update_check.check(root, self.fetch_with(urllib.error.URLError('offline')))['available'])
        self.assertFalse(update_check.check(root, self.fetch_with(
            urllib.error.HTTPError('u', 404, 'Not Found', {}, None)))['available'])

    def test_no_git_folder(self):
        self.assertFalse(update_check.check(self.tmp.name, self.fetch_with({}))['available'])
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
