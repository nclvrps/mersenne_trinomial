# A work ledger for CPU and GPU workers (design note, round 11)

Gord asked for ideas for something to replace `mersenne_trinomial.py`.
CPU and GPU processes should advance together through one range of s,
each at its own speed. Sometimes the GPU is absent and every worker is a
CPU; sometimes the workers run independently in disjoint ranges. Workers
might occasionally resume each other's checkpoints, but at
r = 136279841 a checkpoint is 550 MB or more, too big to keep many.

This note is a proposal to discuss. Nothing in it is implemented yet. The
facts it rests on come first, because two of them change the picture.

---------------------------------------------------------------------------
## 1. Facts that shape the design

**1.1 What a later stage needs from an earlier one is a number.** All the
scan of s establishes is "no factor of degree ≤ d". Continuing from d+1
needs the h-table at d+1. Building it costs about m·d squarings (tsfactor's
init: `Exponentiation took ...`). For the CPU→GPU hand-over at d ≈ 280
that is 34 × 280 squarings × 0.2 ms ≈ 2 s on the GPU, against ~120 s for
the GPU's stage of that survivor. So the hand-over needs no checkpoint. The
remnants file's "s u 279" carries everything. Checkpoints matter only
inside a stage:
* for an interruption in the middle of an interval;
* for deep survivors, where m·d squarings are expensive: 11 min on the
  GPU at d = 10^5, about 2 h at 10^6.

Such survivors are best kept by the worker that has them. A compact way to
move a deep survivor between machines would be a 17 MB record, x^(2^d)
alone. The h-table can be rebuilt from it with about m²/2 products (GPU
m = 34: ~2 min, independent of d). That is a possible tsfactor feature,
listed in §2.9.

**1.2 Where the time goes, per survivor, on Gord's machines** (round 10–11
logs):
* **CPU stage** (tsfactor_cpu, skip 55 → 279): ~75 s on one Zen 2 core
  (round 10, measured); ~73 s expected with round 11's exact maxd.
* **GPU stage** (tsfactor, from 263 to the factor): 120–200 s for typical
  survivors. The GPU is ~9x faster per degree (13 ms against 120 ms), but
  its gcd (18 s) dominates short scans.
* **Deep survivors cost days.** s = 12338357 (primitive) was scanned to
  r/3 = 45.4 million degrees: on the order of a week of GPU time at
  ~13 ms per degree. irred_gpu should prove the same in ~4–8 h; CPU irred
  takes 2.3–2.7 days.

**1.3 When irred beats scanning.** Say the range holds about 3 irreducible
trinomials among its 6.8·10^7 values of s. With the survival law
S(k) ≈ (16/9)/k (§3), the chance that a survivor at degree k is irreducible
is about 2.5·10^-8·k:
* 0.25% at k = 10^5;
* 2.5% at 10^6;
* 10% at 4·10^6.

Running irred (~6 h on the GPU) saves the rest of the scan to r/3 (~7 days)
only when the trinomial is irreducible. So irred pays off once that chance
exceeds 6 h / 7 d ≈ 3.7%, i.e. from k ≈ 1.5·10^6. About 80 survivors of the
whole range get that far. If irred says "reducible", the factor still has
to be found by scanning, and nothing was saved. Above all, irred delivers a
primitive trinomial days earlier.

**1.4 Results are deterministic and every program checks itself.** A
result written twice for the same s must be identical. A disagreement
points to a fault an error check missed, or to a bug. It never happens in
normal operation, so the ledger can treat it as an alarm.

---------------------------------------------------------------------------
## 2. The proposal: a ledger and workers

### 2.1 The ledger

One SQLite file per r, e.g. `r136279841.db`, on the local disk of the
machine where the workers run. It uses WAL mode, and every update is one
short transaction. SQLite comes with Python, so there is nothing to
install. Tables:

    survivor(s INTEGER PRIMARY KEY,
             cleared INTEGER,     -- no factor of degree <= cleared is known
             result TEXT,         -- NULL, "d p<hex>", "primitive",
                                  --   "irreducible", "r<lo>-<hi>"
             irred TEXT,          -- NULL, "irreducible", "x<hex>"/"y<hex>"
             claimed_by TEXT, lease_until REAL, updated REAL)
    event(id INTEGER PRIMARY KEY, t REAL, worker TEXT, s INTEGER,
          kind TEXT, line TEXT)   -- every result line received, verbatim
    worker(id TEXT PRIMARY KEY, kind TEXT, host TEXT, program TEXT,
           started REAL, last_seen REAL)

The ledger is filled by importing:
* coarse_sieve's `outNN.survivors.txt` (cleared = NN);
* any existing `results.txt`, `remnants.txt` and `remnants.results.txt`
  (migration, §2.8).

It is never the only record. Each tsfactor process still appends to its
own `PREFIX.results.txt` with fsync, and the ledger can be rebuilt from
those files and the survivors file. Re-importing is idempotent.

### 2.2 Workers

A worker is a small Python loop around one program process:
`tsfactor_cpu`, `tsfactor` or `irred_gpu`. On each pass it:

1. **Claims a batch.** In one transaction it picks survivors by its
   policy (§2.3) and sets `claimed_by` and `lease_until`. A tsfactor batch
   holds survivors with the same `cleared`, because `--skip` is per
   process. The batch is sized for 20–60 minutes of work, so tuning (4 s
   on a CPU, 10 s on the GPU) is amortised.
2. **Runs the program** with `--skip cleared`, its own `--out`, and the
   batch on stdin. It reads the result lines as they appear and writes each
   one to the ledger at once (`result`, `cleared`, an `event` row).
3. **Renews the lease** every few minutes (a heartbeat). If a worker dies,
   its claims expire and become available again. At worst, another worker
   repeats the dead one's current interval.
4. **Exit codes.** Exit 3 (persistent fault) stops the worker and leaves a
   message in the ledger. A signal makes the worker forward SIGTERM to the
   program; the program checkpoints, and the worker releases the
   survivors it did not start.

There are as many CPU workers as cores (one process per core, each with
its own `--out`), plus one GPU worker per card. The machine can run
whatever mix it has.

### 2.3 Policy: who takes what

There are two stage boundaries:
* **D1**, the CPU stage's end (now 279; it could become a tuning result);
* **D_irr**, where deep survivors go to irred (§1.3: ~1.5·10^6, configurable,
  off by default).

The preferences:

* **CPU worker:** the lowest unclaimed s with `cleared < D1` and no
  result. It scans to D1. If there is none, config decides: stop, idle, or
  take GPU-stage work slowly. The last is useful with no GPU at all.
* **GPU worker:**
  1. the lowest unclaimed s with `D1 ≤ cleared` and no result (scan to
     r/3, or to D_irr);
  2. if there is none, CPU-stage work (the GPU is faster at it too, but
     it should not starve the CPU workers);
  3. if irred is enabled, an s with `cleared ≥ D_irr` and no irred result.
* **Lowest s first** in every stage. The ledger can then report a frontier
  per stage: "every s ≤ S has passed D1", "every s ≤ S' is resolved". That
  is the wavefront Gord asked for, and a range can be closed off by such a
  number.

**Disjoint ranges** are just separate ledgers, or one ledger with a range
filter per worker, as `--range` does now.

### 2.4 Conflicts and audits

A second result for an s that already has one is compared with it:
* identical: recorded as an event only;
* different: an alarm. The s is set aside for a third opinion, and the
  wrapper stops if it happens twice.

Optionally, the ledger hands a small random fraction of finished survivors
(say 1%) to a worker of the other kind for a repeat. That is cross-checking
by construction, at 1% cost.

### 2.5 Checkpoints

They stay local. tsfactor names them `PREFIX_r<r>_s<s>.ckpt` since round
10, and the file holds r, s and skip, but not the prefix. A worker on the
same machine can therefore take over a survivor whose lease expired:
1. the ledger records where the old worker's checkpoint is (host, path,
   degree);
2. the new worker's wrapper renames the files to its own prefix before
   starting the program.

A disk budget, say the newest N checkpoints per machine, deletes the rest.
Losing one costs at most a redone interval.

Across machines, the 17 MB x^(2^d) record of §1.1 would be the thing to
move, if it is ever implemented.

### 2.6 irred in the pipeline

The GPU worker can switch programs per batch:
1. It runs `irred_gpu r s` on a survivor past D_irr.
2. It stores the verdict and residue in `irred`.
3. If irreducible, it sets `result = primitive`.
4. Otherwise the survivor goes back to the scan queue: a factor exists, so
   its scan continues later at low priority.

CPU irred (2.3–2.7 days per s) is an option for idle CPU workers. The program
is the same, so irred_gpu and irred can resume each other's checkpoints
for odd s < r/2.

### 2.7 More than one machine

SQLite must not be shared over NFS. There are two simple options:
* **(a) One ledger per machine, each with its own range.** Merge the
  results files afterwards by import. This is what happens now, and it is
  the right model for cloud batches: hand a VM a range, and collect its
  results file.
* **(b) A coordinator.** A tiny HTTP service (Python's http.server + the
  same SQLite file) on the home machine, with workers claiming and
  reporting over an SSH tunnel. This is worth building only if cloud
  workers should join the same wavefront live.

I would start with (a).

### 2.8 Migration

`ledger import` reads the survivors file, `results.txt` ("s d p<hex>"),
`remnants.txt` ("s u maxd") and the tsfactor results files ("s d p..",
"s u", "s primitive"). It reports conflicts and the frontier. The old
script can then retire.

### 2.9 Program changes that would make it cleaner (tsfactor)

1. **Per-line skip:** input lines "s skip [maxd]", so one process can take
   survivors with different `cleared`.
2. **Stream mode:** read s lines as they arrive, not only after EOF. One
   long-lived process per worker then needs no batches and tunes once.
3. **The degree reached on "u" lines:** `s u 279`. Round 11's exact-maxd
   block size makes this equal to --maxd in the CPU stage. The GPU stage
   can stop elsewhere, for example at D_irr or under -Z.
4. **Rebuilding the h-table from x^(2^d)** with ~m²/2 products (§1.1), to
   move deep survivors cheaply.

### 2.10 Questions for Gord

* Will cloud workers be wanted in the same live wavefront (2.7 b), or are
  ranges per machine enough (2.7 a)?
* Should D1 stay a fixed setting (279), or follow measured speeds? With
  the GPU 9x faster per degree, D1 mainly balances how long the CPU and
  GPU queues are.
* Should deep survivors go to irred automatically, and from what degree?
* Is a 1% cross-check audit wanted?
* Python only, as now? (I would keep it to the standard library.)

---------------------------------------------------------------------------
## 3. Survival law: does it depend on 2^r − 1 being prime?

Gord wondered whether to test non-Mersenne prime r. I ran it in the sandbox
(`tsfactor/experiments/survlaw/`). For every s in [1, (r−1)/2] the program
finds the degree of the least irreducible factor of x^r + x^s + 1 (NTL,
incremental distinct-degree factorisation). That covers:
* 9 Mersenne exponents (3217 … 23209);
* 16 other primes near 9000 and 19000, four of each residue mod 8;
* 168,000 trinomials in all.

k·S(k), where S(k) is the fraction with no factor of degree ≤ k, pooled:

    k =                          2      4      8     16     32     64    128    256    512
    Mersenne, r = ±1 mod 8     1.333  1.524  1.666  1.683  1.718  1.765  1.795  1.795  1.778
    other primes, r = ±1 mod 8 1.333  1.524  1.667  1.738  1.751  1.778  1.767  1.751  1.872
    Mersenne, r = ±3 mod 8     1.333  1.523  1.661  1.728  1.754  1.753  1.739  1.837  1.826
    other primes, r = ±3 mod 8 1.333  1.524  1.662  1.708  1.728  1.718  1.734  1.696  1.637

* **The law is the same for every class.** k·S(k) rises from 4/3 at k = 2
  to about 16/9 = 1.778 from k ≈ 32 on, within the statistical noise. That
  noise grows with k, because few trinomials survive.
* **Whether 2^r − 1 is prime plays no part.** That is as expected: the
  degrees of the factors depend on the orders of x modulo them, which have
  nothing to do with 2^r − 1.
* **r mod 8 matters only far out.** For r ≡ ±3 mod 8, Swan's theorem
  makes every trinomial reducible (0 irreducible ones in the data). For
  r ≡ ±1 mod 8 the few irreducible trinomials add a constant to S(k), which
  shows only when k is a sizeable fraction of r. That is why the per-r
  columns rise at k = 1024 for r ≈ 9000. At r = 136279841 it would show
  only beyond k ≈ 10^7.

**For the programs:** the DP schedule uses only ratios S(k')/S(k) = k/k',
so it holds for any prime r. The one r-specific thing tsfactor had was the
word "primitive" for a survivor with no factor up to the bound. Round 11
prints "irreducible" instead unless r is one of the 52 known Mersenne
exponents. irreducible means primitive only when 2^r − 1 is prime.
Gord need not spend machine time on the question.
