"""Fixed timing investigation; run at repository root after all builds finish.

Usage: python3 collect.py <original-cinder> <current-cinder> <probe> <new-output-dir>
The probe's recordings change overhead. This is not a speedup comparison.
Every process outcome is retained; a rejected measurement is never replaced.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def identity(path):
    p = Path(path)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def command(args):
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=60)
    return dict(command=list(map(str, args)), started_utc=started,
                exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)


original, current, probe, destination = map(Path, sys.argv[1:])
destination.mkdir()  # Do not overwrite an earlier collection.
cases = dict(tiny=('tests/fixtures/tiny/tiny.cnd', 'tests/fixtures/tiny/inputs/worked.f32'),
             mnist=('examples/mnist/mnist.cnd', 'tests/fixtures/mnist/inputs/00000.f32'))
report = dict(checkout=command(['git', 'rev-parse', 'HEAD']),
              source_status=command(['git', 'status', '--short']),
              source_diff=command(['git', 'diff', '--', 'src', 'include']),
              os=command(['uname', '-a']), compiler=command(['cc', '--version']),
              load_before=os.getloadavg(), artifacts=[identity(p) for p in (original, current, probe)],
              fixtures=[identity(p) for pair in cases.values() for p in pair],
              protocol='20 attempts/tool for tiny at 100000 iterations; 3/tool for MNIST at 1000; '
                       'three collections of 1000000 adjacent raw-clock pairs; no retries or filtering; '
                       'active desktop, unpinned cores, no concurrent agent builds/tests',
              trials=[])


def record(tool, case, trial, args):
    entry = dict(tool=tool, case=case, trial=trial, **command(list(map(str, args))))
    report['trials'].append(entry)
    (destination/'observations.json').write_text(json.dumps(report, indent=2)+'\n')
    print(tool, case, trial, 'exit', entry['exit_status'], flush=True)


for trial in range(1, 21):
    model, inputs = cases['tiny']
    for tool, executable in (('original', original), ('current', current), ('probe', probe)):
        args = [executable, model, inputs, '100000'] if tool == 'probe' else [
            executable, 'bench', model, inputs, '--iterations', '100000']
        record(tool, 'tiny', trial, args)
for trial in range(1, 4):
    model, inputs = cases['mnist']
    for tool, executable in (('original', original), ('current', current), ('probe', probe)):
        args = [executable, model, inputs, '1000'] if tool == 'probe' else [
            executable, 'bench', model, inputs, '--iterations', '1000']
        record(tool, 'mnist', trial, args)
    record('probe', 'clock', trial, [probe, 'clock'])
report['load_after'] = os.getloadavg()
assert report['artifacts'] == [identity(p) for p in (original, current, probe)]
assert report['fixtures'] == [identity(p) for pair in cases.values() for p in pair]
(destination/'observations.json').write_text(json.dumps(report, indent=2)+'\n')
