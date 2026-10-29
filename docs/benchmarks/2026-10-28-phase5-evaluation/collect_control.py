"""Repeat the M5 baseline without changing code; run from the repository root.

Usage: python3 collect_m5_control.py <Release-build-dir> <new-output-dir>
The output directory must be ignored by Git. No retries or failure filtering.
"""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

root = Path.cwd()
build, output = map(Path, sys.argv[1:])
baseline_path = Path('docs/benchmarks/2026-10-28-apple-m4/baseline.json')
baseline = json.loads(baseline_path.read_text())

def command(args):
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return dict(argv=args, exit_status=p.returncode, stdout=p.stdout, stderr=p.stderr)

def git(*args):
    r = command(['git', *args])
    assert r['exit_status'] == 0, r
    return r['stdout'].strip()

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def identity(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def snapshot():
    return dict(at_utc=stamp(), load_averages_1_5_15_minutes=list(os.getloadavg()),
                power=command(['pmset', '-g', 'batt']),
                power_settings=command(['pmset', '-g', 'custom']),
                thermal=command(['pmset', '-g', 'therm']))

assert git('status', '--porcelain', '--untracked-files=all') == '', 'Use a clean checkout.'
source_paths = ['src', 'include', 'tests', 'CMakeLists.txt', '.github', 'examples', 'python']
assert git('diff', baseline['commit'], 'HEAD', '--', *source_paths) == ''
executable = identity(build/'cinder')
assert executable['sha256'] == baseline['executable']['sha256'], 'This control requires the identical baseline executable.'
assert executable['bytes'] == baseline['executable']['bytes']
for case in baseline['fixtures']:
    for kind in ['model', 'input']:
        assert identity(case[kind]['path']) == case[kind]
instrumentation = [k for k in ('DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'MallocStackLogging',
    'MallocScribble', 'MallocGuardEdges', 'ASAN_OPTIONS', 'UBSAN_OPTIONS') if k in os.environ]
assert not instrumentation, instrumentation
output.mkdir()  # Never overwrite an earlier collection.
assert git('status', '--porcelain', '--untracked-files=all') == ''
report = {k: baseline[k] for k in ['machine', 'fixtures', 'build']}
report.update(schema_version=1, purpose='Unchanged-executable control; not an optimized after-build.',
    checkout_commit=git('rev-parse', 'HEAD'), compiled_source_commit=baseline['commit'],
    git_tree=git('rev-parse', 'HEAD^{tree}'), clean_before_trials=True,
    unchanged_source_paths=source_paths, baseline=identity(baseline_path), executable=executable,
    operating_system=command(['sw_vers']), compiler=command(['/usr/bin/cc', '--version']),
    cmake=command(['cmake', '--version']), sdk_version=command(['xcrun', '--show-sdk-version']),
    protocol=dict(trials_per_model=5, measured_iterations_per_trial=100000, warmup_iterations=10,
        order='tiny then mnist, repeated five times', pause_after_each_trial_seconds=1,
        clock='CLOCK_MONOTONIC_RAW', runtime_threads=1, process_per_trial=True,
        core_affinity='not pinned', frequency_control='system managed',
        background_conditions='active desktop; no agent builds/tests/leak checks during collection; other applications left running',
        filtering='none; every attempt retained; no retries', instrumentation_environment_keys_present=instrumentation),
    environment_before=snapshot(), trials=[])
raw_path = output/'raw-trials.txt'
raw_path.write_text('Cinder M5 unchanged-executable control\nCheckout commit: '+report['checkout_commit']+'\n\n')
labels = ['Model', 'Iterations', 'Warm-up iterations', 'Clock', 'Clock resolution (ns)',
          'Mean latency (us)', 'Minimum latency (us)', 'Maximum latency (us)', 'Throughput (inferences/s)']

def persist():
    (output/'control.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

persist()
for trial in range(1, 6):
    for case in report['fixtures']:
        args = [str(build.resolve()/'cinder'), 'bench', case['model']['path'], case['input']['path'], '--iterations', '100000']
        started, load_before = stamp(), list(os.getloadavg())
        result = command(args)
        ended = stamp()
        args[0] = './'+str(build/'cinder')  # Normalize only the displayed path.
        entry = dict(model=case['name'], trial=trial, started_at_utc=started, finished_at_utc=ended,
            load_averages_before=load_before, load_averages_after=list(os.getloadavg()), command=args,
            exit_status=result['exit_status'], stderr=result['stderr'], valid=False)
        with raw_path.open('a') as raw:
            raw.write(f"Trial {trial} / {case['name']}\nStarted UTC: {started}\nFinished UTC: {ended}\n")
            raw.write('Command: '+' '.join(args)+'\n')
            raw.write(f"Exit status: {result['exit_status']}\nStdout:\n{result['stdout']}")
            raw.write('Stderr: '+repr(result['stderr'])+'\n\n')
        report['trials'].append(entry)
        persist()
        if result['exit_status'] != 0:
            print(f"{case['name']} trial {trial}/5: FAILED; {result['stderr'].strip()}", flush=True)
            time.sleep(1)
            continue
        assert not result['stderr']
        lines = result['stdout'].splitlines()
        assert len(lines) == len(labels) and result['stdout'].endswith('\n')
        values = {}
        for line, label in zip(lines, labels):
            assert line.startswith(label+': ')
            values[label] = line[len(label)+2:]
        assert values['Model'] == Path(case['model']['path']).name
        assert int(values['Iterations']) == 100000 and int(values['Warm-up iterations']) == 10
        assert values['Clock'] == 'monotonic'
        metrics = dict(iterations=100000, warmups=10, resolution_ns=int(values['Clock resolution (ns)']),
            mean_us=float(values['Mean latency (us)']), min_us=float(values['Minimum latency (us)']),
            max_us=float(values['Maximum latency (us)']), inferences_per_second=float(values['Throughput (inferences/s)']))
        assert all(math.isfinite(v) and v > 0 for v in metrics.values())
        assert metrics['min_us'] <= metrics['mean_us'] <= metrics['max_us']
        assert abs(metrics['mean_us']*metrics['inferences_per_second']-1e6) <= .02
        entry.update(metrics=metrics, valid=True)
        persist()
        print(f"{case['name']} trial {trial}/5: mean={metrics['mean_us']:.9g} us; throughput={metrics['inferences_per_second']:.9g}/s", flush=True)
        time.sleep(1)
report['environment_after'] = snapshot()
assert git('rev-parse', 'HEAD') == report['checkout_commit']
assert git('status', '--porcelain', '--untracked-files=all') == ''
report['clean_after_trials'] = True
assert identity(build/'cinder') == executable
for case in report['fixtures']:
    for kind in ['model', 'input']:
        assert identity(case[kind]['path']) == case[kind]
report['summaries'] = {}
for case in report['fixtures']:
    attempts = [t for t in report['trials'] if t['model'] == case['name']]
    valid = [t['metrics'] for t in attempts if t['valid']]
    summary = dict(attempted_trials=len(attempts), valid_trials=len(valid), failed_trials=len(attempts)-len(valid),
        total_measured_calls_in_valid_trials=sum(t['iterations'] for t in valid))
    if valid:
        summary.update(median_trial_mean_us=statistics.median(t['mean_us'] for t in valid),
            range_of_trial_means_us=[min(t['mean_us'] for t in valid), max(t['mean_us'] for t in valid)],
            median_trial_throughput=statistics.median(t['inferences_per_second'] for t in valid),
            fastest_individual_call_us=min(t['min_us'] for t in valid), slowest_individual_call_us=max(t['max_us'] for t in valid))
    report['summaries'][case['name']] = summary
# Omit only the final separator blank line; all captured process output is intact.
raw_path.write_text(raw_path.read_text()[:-1])
report['raw_trials'] = identity(raw_path)
persist()
print('All ten attempts retained; source, executable, and fixtures unchanged.', flush=True)
