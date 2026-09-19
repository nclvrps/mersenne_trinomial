#!/usr/bin/env python3
import os
import sys
import glob
import re
import subprocess
import threading
import signal
import time
import argparse

# ==========================================
# Hardcoded constants
# ==========================================
r_val = 136279841

# ==========================================
# Global state
# ==========================================
k_val = None
config_params = {
    'm': 14,
    'q': 15,
    'batchsize': 10,
    'instances': 1,
    'sched_g_global': None,
    'sched_g_specific': {},
    'tsfactor_env': None
}
ranges = []

# State for survivors and dispatching
unprocessed_survivors = []  
pending_survivors = []      
state_lock = threading.Lock()
file_lock = threading.Lock()

# Execution control
drawdown_mode = False
sigterm_received = False
sigint_count = 0

# ==========================================
# Helper Functions
# ==========================================
def make_filelist(basename: str):
    pattern = re.compile(
        rf"""
        ^                       
        (?:                     
            .*?                 
            [._]                
        )?                      
        {basename}              
        (?:                     
            \s                  
            \( \d+ \)           
        )?                      
        \.txt                   
        $                       
        """,
        re.VERBOSE,
    )
    return [f for f in os.listdir(".") if pattern.search(f)]

def get_linux_physical_cores():
    try:
        cores = set()
        with open("/proc/cpuinfo", "r") as f:
            current_core = {}
            for line in f:
                if not line.strip():
                    if "physical id" in current_core and "core id" in current_core:
                        cores.add((current_core["physical id"], current_core["core id"]))
                    current_core = {}
                elif ":" in line:
                    key, val = line.split(":", 1)
                    current_core[key.strip()] = val.strip()
        return len(cores) if cores else 1
    except Exception:
        return 1

def parse_num(s):
    s = s.strip()
    try:
        if 'M' in s:
            return int(float(s.replace('M', '')) * 1000000)
        if 'k' in s:
            return int(float(s.replace('k', '')) * 1000)
        return int(s)
    except ValueError:
        return None

def parse_range_line(val):
    val = val.strip()
    if '-' in val:
        parts = val.split('-')
        if len(parts) == 2:
            start = parse_num(parts[0])
            end = parse_num(parts[1]) - 1
            if start is not None and end is not None:
                return (start, end)
    
    if 'k' not in val and 'M' not in val:
        n = parse_num(val)
        if n is not None:
            return (n, n)
    else:
        multiplier_zeros = 6 if 'M' in val else 3
        v = val.replace('M', '').replace('k', '')
        
        if '.' in v:
            int_part, frac_part = v.split('.', 1)
        else:
            int_part, frac_part = v, ''
            
        rem_zeros = multiplier_zeros - len(frac_part)
        if rem_zeros >= 0:
            start_str = int_part + frac_part + ('0' * rem_zeros)
            end_str = int_part + frac_part + ('9' * rem_zeros)
            return (int(start_str), int(end_str))
            
    return None

def check_overlaps(rngs):
    sorted_r = sorted(rngs)
    for i in range(len(sorted_r) - 1):
        if sorted_r[i][1] >= sorted_r[i+1][0]:
            print(f"WARNING: Ranges overlap: {sorted_r[i]} and {sorted_r[i+1]}", file=sys.stderr)

def read_config():
    global config_params, ranges
    config_path = "mersenne_trinomial.config"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    new_ranges = []
    new_params = {
        'm': 14,
        'q': 15,
        'batchsize': 10,
        'instances': 1,
        'sched_g_global': None,
        'sched_g_specific': {},
        'tsfactor_env': None
    }
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if ':' in line and '=' not in line:
                key, val = line.split(':', 1)
            elif '=' in line and ':' not in line:
                key, val = line.split('=', 1)
            elif ':' in line and '=' in line:
                first_sep = min(line.find(':'), line.find('='))
                key, val = line[:first_sep], line[first_sep+1:]
            else:
                continue
                
            key = key.strip().lower()
            val = val.strip()
            
            if key == 'range':
                r_tuple = parse_range_line(val)
                if r_tuple:
                    new_ranges.append(r_tuple)
            elif key in ['instances', 'm', 'q', 'batchsize']:
                try:
                    new_params[key] = int(val)
                except ValueError:
                    pass
            elif key in ['sched-g', 'sched_g']:
                parts = [p.strip() for p in val.split(',')]
                if len(parts) == 2:
                    try:
                        new_params['sched_g_specific'][int(parts[0])] = parts[1]
                    except ValueError:
                        pass
                elif len(parts) == 1:
                    new_params['sched_g_global'] = parts[0]
            elif key in ['tsfactor_env', 'tsfactor-env']:
                cleaned_env = val.removeprefix("export ").strip()
                if cleaned_env:
                    try:
                        new_params['tsfactor_env'] = {
                            k: v for k, v in (item.split("=", 1) for item in cleaned_env.split())
                        }
                    except ValueError:
                        print(f"WARNING: Malformed environment string format in config: '{val}'", file=sys.stderr)

    physical_cores = get_linux_physical_cores()
    if new_params['instances'] > physical_cores:
        print(f"WARNING: 'instances' ({new_params['instances']}) exceeds physical cores ({physical_cores}). Reducing to {physical_cores}.", file=sys.stderr)
        new_params['instances'] = physical_cores
    
    check_overlaps(new_ranges)
    return new_params, new_ranges

def filter_pending_survivors():
    global pending_survivors, unprocessed_survivors, ranges
    if not ranges:
        pending_survivors = unprocessed_survivors.copy()
        return

    filtered = []
    for s in unprocessed_survivors:
        if any(start <= s <= end for start, end in ranges):
            filtered.append(s)
            
    pending_survivors = filtered

# ==========================================
# Shared Steps 1, 2, and 3
# ==========================================
def fetch_base_survivors():
    global k_val, config_params, ranges

    files = glob.glob("out*.survivors.txt") + glob.glob(f"{r_val}_*.survivors.txt")
    highest_k = -1
    survivors_file = None
    
    for f in files:
        m = re.match(r'^(?:out|' + str(r_val) + r'_)(\d{2})\.survivors\.txt$', f)
        if m:
            k = int(m.group(1))
            if k > highest_k:
                highest_k = k
                survivors_file = f
                
    if not survivors_file:
        print("Error: No valid survivors file found.", file=sys.stderr)
        sys.exit(1)
        
    k_val = highest_k
    print(f"Notice: Using k={k_val} from {survivors_file}", file=sys.stderr)
    
    survivors_set = []
    with open(survivors_file, 'r') as f:
        prev_s = -1
        line_num = 0
        for line in f:
            line_num += 1
            line = line.strip()
            if not line: continue
            
            try:
                s = int(line)
                if s <= 0:
                    print(f"Warning: Found non-positive integer {s} on line {line_num} in {survivors_file}", file=sys.stderr)
                if s <= prev_s:
                    print(f"Warning: Values not strictly ascending on line {line_num} in {survivors_file}", file=sys.stderr)
                if s > r_val // 2:
                    print(f"Warning: Value {s} > r//2 on line {line_num} in {survivors_file}", file=sys.stderr)
                    
                survivors_set.append(s)
                prev_s = s
            except ValueError:
                print(f"Warning: Non-integer value on line {line_num} in {survivors_file}", file=sys.stderr)
                
    if not survivors_set:
        print("Error: Survivors set is empty.", file=sys.stderr)
        sys.exit(1)

    config_params, ranges = read_config()
    
    if ranges:
        survivors_set = [s for s in survivors_set if any(start <= s <= end for start, end in ranges)]
    
    if not survivors_set:
        return [] 

    remove_s = set()
    result_files = make_filelist("results")
    
    for rf in result_files:
        with open(rf, 'r') as f:
            for line in f:
                line = line.strip()
                match_sdp = re.match(r"^(\d+)\s+(\d+)\s+p[0-9a-f]+$", line)
                if match_sdp:
                    s = int(match_sdp.group(1))
                    d = int(match_sdp.group(2))
                    if not (1 <= s <= r_val // 2) or not (2 <= d <= r_val // 3):
                        print(f"Error: Invalid bounds in result file {rf}: s={s}, d={d}", file=sys.stderr)
                        sys.exit(1)
                    remove_s.add(s)
                    continue
                    
                match_rem = re.match(r"^(\d+)\s+(primitive|irreducible|u)$", line)
                if match_rem:
                    s = int(match_rem.group(1))
                    if not (1 <= s <= r_val // 2):
                        print(f"Error: Invalid bounds in result file {rf}: s={s}", file=sys.stderr)
                        sys.exit(1)
                    remove_s.add(s)
                    
    survivors_set = [s for s in survivors_set if s not in remove_s]
    return survivors_set

# ==========================================
# Signal Handlers
# ==========================================
def handle_sigint(sig, frame):
    global drawdown_mode, sigint_count
    sigint_count += 1
    if sigint_count == 1:
        drawdown_mode = True
        print("\n[Ctrl-C received] Entering drawdown mode. No new subprocesses will be started. "
              "Waiting for existing ones to finish. Press Ctrl-C again to terminate immediately.", file=sys.stderr)
    else:
        print("\n[Second Ctrl-C received] Terminating immediately.", file=sys.stderr)
        sys.exit(1)

def handle_sigterm(sig, frame):
    global sigterm_received
    print("\n[SIGTERM received] Terminating subprocesses immediately.", file=sys.stderr)
    sigterm_received = True

# ==========================================
# Worker Thread (factor_ec mode)
# ==========================================
def worker_thread():
    global pending_survivors
    
    while not sigterm_received:
        with state_lock:
            if drawdown_mode or not pending_survivors:
                return
                
            batch_size = config_params['batchsize']
            batch = pending_survivors[:batch_size]
            pending_survivors = pending_survivors[batch_size:]
            
            # The O(N^2) list comprehension filtering unprocessed_survivors has been entirely 
            # removed here, as it was redundant and caused single-core performance stalls.
            
            m = config_params['m']
            q = config_params['q']
            local_k = k_val
        
        if not batch:
            return
            
        computed_maxd = local_k + m * q * 1 * (1 + 1) // 2

        cmd = ["./factor_ec", "-v", "-p", "-f", "1", "-s0", "101", "-s1", "100",
               "-k", str(local_k), "-skip", str(local_k), "-m", str(m), "-q", str(q), "-maxd", str(computed_maxd), str(r_val)]
               
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                    start_new_session=True)
        except FileNotFoundError:
            print("Error: 'factor_ec' executable not found in PATH.", file=sys.stderr)
            sys.exit(1)
            
        try:
            for s in batch:
                proc.stdin.write(f"{s}\n")
            proc.stdin.close()
        except IOError:
            pass

        curr_maxd = computed_maxd

        for line in proc.stdout:
            if sigterm_received:
                proc.terminate()
                break
                
            line = line.strip()
            if not line: continue
            
            match_interval = re.match(r"^Interval\s+(\d+)\.\.(\d+):$", line)
            if match_interval:
                curr_mind = int(match_interval.group(1))
                if curr_mind != k_val + 1:
                    print(f"WARNING: Expected start degree ({k_val + 1}) does not match Interval start ({curr_mind})", file=sys.stderr)
                curr_maxd = int(match_interval.group(2))
                if curr_maxd != computed_maxd:
                    print(f"WARNING: Calculated maxd ({computed_maxd}) does not match Interval maxd ({curr_maxd})", file=sys.stderr)
                continue
                
            match_sp = re.match(r"^squares/products took (\d+\.\d+) ", line)
            if match_sp:
                curr_sp = float(match_sp.group(1))
                print(f"   ### squares/products took {curr_sp}", file=sys.stderr)
                continue

            match_gcd = re.match(r"^gcd took (\d+\.\d+)$", line)
            if match_gcd:
                curr_gcd = float(match_gcd.group(1))
                print(f"   ### gcd took {curr_gcd}", file=sys.stderr)
                continue
                
            match_sdp = re.match(r"^(\d+)\s+(\d+)\s+p[0-9a-f]+$", line)
            if match_sdp:
                with file_lock:
                    with open("results.txt", "a") as rf:
                        rf.write(line + "\n")
                continue
                
            match_remnant = re.match(r"^(\d+)\s+(primitive|irreducible|u)$", line)
            if match_remnant:
                s_val = match_remnant.group(1)
                with file_lock:
                    with open("remnants.txt", "a") as rmf:
                        rmf.write(f"{s_val} u {curr_maxd}\n")
                continue
                
        proc.wait()

# ==========================================
# Execution Modes
# ==========================================
def run_factor_ec_mode():
    global unprocessed_survivors, pending_survivors

    import shutil
    if not shutil.which("./factor_ec"):
        print("Error: 'factor_ec' executable not found.", file=sys.stderr)
        sys.exit(1)

    survivors_set = fetch_base_survivors()
    if not survivors_set:
        print("Notice: All ranges completed (survivors set empty).", file=sys.stderr)
        sys.exit(0)

    remnant_files = make_filelist("remnants")
    remove_s = set()
    
    for rf in remnant_files:
        with open(rf, 'r') as f:
            for line in f:
                line = line.strip()
                match_rem = re.match(r"^(\d+)\s+u\s+(\d+)$", line)
                if match_rem:
                    s = int(match_rem.group(1))
                    maxd = int(match_rem.group(2))
                    if not (1 <= s <= r_val // 2) or not (2 <= maxd <= r_val // 3):
                        print(f"Error: Invalid bounds in remnants file {rf}: s={s}, maxd={maxd}", file=sys.stderr)
                        sys.exit(1)
                    remove_s.add(s)

    survivors_set = [s for s in survivors_set if s not in remove_s]
    if not survivors_set:
        print("Notice: All ranges completed (survivors set empty after remnants filter).", file=sys.stderr)
        sys.exit(0)

    unprocessed_survivors = survivors_set
    filter_pending_survivors()
    
    active_threads = []
    try:
        while not sigterm_received:
            active_threads = [t for t in active_threads if t.is_alive()]
            with state_lock:
                has_work = bool(pending_survivors)
                target_instances = config_params['instances']
                
            if drawdown_mode:
                if not active_threads:
                    break
            elif has_work and len(active_threads) < target_instances:
                t = threading.Thread(target=worker_thread, daemon=True)
                t.start()
                active_threads.append(t)
            elif not has_work and not active_threads:
                break
            time.sleep(0.1)
    finally:
        pass
        
    print("Notice: All requested processing is complete.", file=sys.stderr)

def run_tsfactor_mode():
    import shutil
    if not shutil.which("./tsfactor"):
        print("Error: './tsfactor' executable not found.", file=sys.stderr)
        sys.exit(1)

    while not sigterm_received:
        if drawdown_mode:
            break

        base_survivors = fetch_base_survivors()
        if not base_survivors:
            print("Notice: Remaining survivors set is empty.", file=sys.stderr)
            sys.exit(0)
            
        base_survivors_set = set(base_survivors)

        remnants_set = set()
        s_maxd_map = {}
        remnant_files = make_filelist("remnants")
        
        for rf in remnant_files:
            with open(rf, 'r') as f:
                for line in f:
                    line = line.strip()
                    match_rem = re.match(r"^(\d+)\s+u\s+(\d+)$", line)
                    if match_rem:
                        s = int(match_rem.group(1))
                        maxd = int(match_rem.group(2))
                        
                        if s in base_survivors_set:
                            if s in s_maxd_map and s_maxd_map[s] != maxd:
                                print(f"Error: Conflicting maxd for s={s}: {s_maxd_map[s]} and {maxd}", file=sys.stderr)
                                sys.exit(1)
                            s_maxd_map[s] = maxd
                            remnants_set.add((s, maxd))

        if not remnants_set:
            print("Notice: Final remnants set is empty. Terminating.", file=sys.stderr)
            sys.exit(0)

        maxd_groups = {}
        for s, maxd in remnants_set:
            maxd_groups.setdefault(maxd, []).append(s)

        for maxd in maxd_groups:
            maxd_groups[maxd].sort()

        sorted_maxd_keys = sorted(maxd_groups.keys(), key=lambda maxd: maxd_groups[maxd][0])

        for maxd in sorted_maxd_keys:
            print(f"Notice: Processing remnants having maxd: {maxd}", file=sys.stderr)

            if sigterm_received or drawdown_mode:
                break

            cmd = ["./tsfactor", str(r_val), "/dev/stdin", "--skip", str(maxd),
                   "--sched", "opt", "--out", "remnants", "--ckpt-mins", "30",
                   "--gcd-threads", "1", "--pend-max", "1", "-v", "-v", "-v"]

            sched_g = config_params['sched_g_specific'].get(maxd)
            if sched_g is None:
                sched_g = config_params['sched_g_global']
                
            if sched_g is not None:
                cmd.extend(["--sched-G", str(sched_g)])
                
            if config_params['tsfactor_env'] is not None:
                ts_env = {**os.environ, **config_params['tsfactor_env']}
            else:
                ts_env = os.environ

            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, bufsize=1, env=ts_env)
                for s in maxd_groups[maxd]:
                    proc.stdin.write(f"{s}\n")
                proc.stdin.close()
                proc.wait()
            except FileNotFoundError:
                print("Error: './tsfactor' executable not found during execution.", file=sys.stderr)
                sys.exit(1)
            except IOError:
                pass 

        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Mersenne Trinomial processing script")
    parser.add_argument("--tsfactor", action="store_true", help="Run in tsfactor mode")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigterm)

    if args.tsfactor:
        run_tsfactor_mode()
    else:
        run_factor_ec_mode()

if __name__ == "__main__":
    main()
