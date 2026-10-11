"""Compile, import, test documented --help, and run protocol tests on CPU."""
import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/cpu.json')
    args = parser.parse_args()
    files = list((ROOT / 'remem').glob('*.py')) + list((ROOT / 'scripts').glob('validate_*.py')) + list((ROOT / 'tests').glob('*.py'))
    for path in files:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    modules = []
    for path in (ROOT / 'remem').glob('*.py'):
        name = f'remem.{path.stem}'
        importlib.import_module(name)
        modules.append(name)
    commands = [['-m', 'remem.data'], ['-m', 'remem.train'], ['-m', 'remem.unlearn'],
                ['-m', 'remem.evaluate'], ['scripts/validate_remote.py'], ['scripts/validate_local.py'],
                ['scripts/validate_model.py']]
    for command in commands:
        completed = subprocess.run([sys.executable, *command, '--help'], cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'python': sys.version.split()[0], 'compiled_files': len(files), 'imported_modules': modules,
              'help_commands': ['python ' + ' '.join(c) + ' --help' for c in commands],
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'transformers',
                            'peft', 'accelerate', 'datasets', 'huggingface-hub', 'Pillow', 'numpy', 'rouge-score')}}
    from remem.utils import save_json
    save_json(ROOT / args.output, report)
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__ == '__main__':
    main()
