/* gecsim.cpp -- fault-injection study of error checking for the multi-level
 * blocking DDF trinomial factor search (model lives in gecsim_core.h).
 *
 * Build:  g++ -O2 -o gecsim gecsim.cpp
 * Run:    ./gecsim            (all experiments)
 *         ./gecsim N          (experiment N only)
 */
#include "gecsim_core.h"

/* Experiment target: a trinomial whose smallest factor has degree 36, reached
   after 3 intervals of q=3 blocks of m=4 -- i.e. 9 blocks, ~150 modular
   squarings, ~12 modular multiplications, 3 gcds.  Small enough that every
   single-fault position can be swept exhaustively. */
static const long R_ = 1279, S_ = 345;
static const long M_ = 4, K0_ = 3, Q_ = 3, MAXD_ = 60;
static const long NFOLD_ = 1024;      /* fold modulus x^1024 + 1 */

struct Outcome { long answer; long detections; bool aborted; };

static Outcome run_once(bool checks, bool gerbicz, long window, FaultModel *fm)
{
  Engine E(R_, S_, NFOLD_, fm, checks);
  ScanResult sr = scan(E, K0_, M_, Q_, MAXD_, gerbicz, window);
  Outcome o; o.answer = sr.found_degree; o.detections = sr.detections;
  o.aborted = sr.aborted;
  return o;
}

/* -------------------------------------------------------------- */
static void exp1_identity()
{
  printf("=== 1. Is  sigma(sum_j h_j^(z)) == sum_j h_j^(z+1)  an identity? ===\n");
  long ms[5] = {2, 3, 4, 6, 8};
  for (int mi = 0; mi < 5; mi++) {
    long m = ms[mi];
    Engine E(R_, S_, NFOLD_, NULL, false);
    std::vector<Poly> h = init_h(E, K0_ + 1, m);
    bool ok = true;
    for (long z = 0; z < 10; z++) {
      Poly S; for (long j = 0; j < m; j++) addto(S, h[(size_t)j]);
      Poly sS = S; for (long l = 0; l < m; l++) sS = E.R.sqrmod(sS);
      Poly Sn;
      for (long j = 0; j < m; j++) {
        for (long l = 0; l < m; l++) h[(size_t)j] = E.R.sqrmod(h[(size_t)j]);
        addto(Sn, h[(size_t)j]);
      }
      if (!equal(sS, Sn)) ok = false;
    }
    printf("   m=%ld, 10 consecutive blocks: %s\n", m, ok ? "HOLDS" : "FAILS");
  }
  printf("\n");
}

/* -------------------------------------------------------------- */
static void exp2_baseline()
{
  printf("=== 2. Fault-free baseline ===\n");
  Ring R0(R_, S_);
  long ref = reference_smallest(R0, MAXD_);
  Engine E(R_, S_, NFOLD_, NULL, true);
  ScanResult sr = scan(E, K0_, M_, Q_, MAXD_, true, 1);
  printf("   x^%ld + x^%ld + 1 : reference smallest factor degree = %ld\n",
         R_, S_, ref);
  printf("   blocked-DDF answer = %ld, spurious detections = %ld\n",
         sr.found_degree, sr.detections);
  printf("   work: %ld modular squarings, %ld modular multiplications\n",
         E.st.sqr_ops, E.st.mul_ops);
  printf("   checker: %ld extra squarings (%.1f%% of squaring work), %ld folds\n",
         E.st.check_sqr_ops,
         100.0 * E.st.check_sqr_ops / (double)E.st.sqr_ops, E.st.fold_ops);
  printf("\n");
}

/* -------------------------------------------------------------- */
static void set_target(FaultModel &fm, int kind, long t)
{
  switch (kind) { case 0: fm.target_sqr = t; break;
                  case 1: fm.target_mul = t; break;
                  case 2: fm.target_horn = t; break;
                  case 3: fm.target_gcd = t; break;
                  case 4: fm.target_idle = t; break; }
}

static void sweep(const char *name, int kind, long nops, long window)
{
  Ring R0(R_, S_);
  long ref = reference_smallest(R0, MAXD_);
  long silent_wrong = 0, loud_wrong = 0, harmless = 0;
  long detected = 0, notflagged = 0, wrong_despite = 0, aborted = 0;

  for (long t = 0; t < nops; t++) {
    { FaultModel fm(99991 + (uint64_t)t); set_target(fm, kind, t);
      Outcome o = run_once(false, false, window, &fm);
      if (fm.injected == 0) continue;
      if (o.answer == ref) harmless++;
      else if (o.answer < 0 || o.answer > ref) silent_wrong++;
      else loud_wrong++; }
    { FaultModel fm(99991 + (uint64_t)t); set_target(fm, kind, t);
      Outcome o = run_once(true, true, window, &fm);
      if (fm.injected == 0) continue;
      if (o.aborted) { aborted++; continue; }
      if (o.detections > 0) detected++; else notflagged++;
      if (o.answer != ref) wrong_despite++; }
  }
  long total = silent_wrong + loud_wrong + harmless;
  printf("   %-24s faults injected = %ld\n", name, total);
  printf("      unchecked: %ld SILENT wrong (factor missed), %ld self-evident, "
         "%ld harmless\n", silent_wrong, loud_wrong, harmless);
  printf("      checked  : %ld detected+recovered, %ld not flagged "
         "(answer right anyway), %ld wrong\n", detected, notflagged, wrong_despite);
  if (aborted) printf("      (%ld aborted)\n", aborted);
}

static void exp3_sweeps()
{
  printf("=== 3. Exhaustive single-fault sweep: one bit flip per run, every\n"
         "       operation of each class targeted in turn ===\n");
  sweep("squaring (h-table)",  0, 400, 1);
  sweep("modular multiply",    1, 40,  1);
  sweep("Horner (build C[z])", 2, 200, 1);
  sweep("gcd result",          3, 10,  1);
  sweep("idle h-table memory", 4, 60,  1);
  printf("\n");
}

/* -------------------------------------------------------------- */
static void exp4_window()
{
  printf("=== 4. Telescoping window w (Gerbicz check every w blocks) vs\n"
         "       detection of squaring faults ===\n");
  Ring R0(R_, S_);
  long ref = reference_smallest(R0, MAXD_);
  long ws[4] = {1, 2, 3, 9};
  for (int wi = 0; wi < 4; wi++) {
    long w = ws[wi];
    long det = 0, wrong = 0, n = 0;
    for (long t = 0; t < 400; t++) {
      FaultModel fm(555 + (uint64_t)t); fm.target_sqr = t;
      Outcome o = run_once(true, true, w, &fm);
      if (fm.injected == 0) continue;
      n++;
      if (o.detections > 0) det++;
      if (!o.aborted && o.answer != ref) wrong++;
    }
    printf("   w=%-2ld  faults=%-4ld detected=%-4ld wrong=%ld   "
           "checker sigma cost ~= %.2f%% of squaring work\n",
           w, n, det, wrong, 100.0 / (double)(w * M_));
  }
  printf("\n");
}

/* -------------------------------------------------------------- */
static void exp5_foldsize()
{
  printf("=== 5. Cyclic fold checksum: can a bit flip survive the fold? ===\n");
  std::mt19937_64 rng(7);
  long ns[4] = {64, 256, 1024, 4096};
  for (int ni = 0; ni < 4; ni++) {
    long n = ns[ni];
    Fold F(n);
    long survivors = 0, bsurv = 0, trials = 100000;
    for (long t = 0; t < trials; t++) {
      Poly e; flipbit(e, (long)(rng() % (uint64_t)(4 * R_)));
      if (!Fold::isnz(F.of(e))) survivors++;
    }
    for (long t = 0; t < trials; t++) {
      Poly e; long base = (long)((rng() % (uint64_t)(4 * R_)) & ~(uint64_t)63);
      uint64_t pat = rng(); if (!pat) pat = 1;
      for (int b = 0; b < 64; b++) if ((pat >> b) & 1) flipbit(e, base + b);
      if (!Fold::isnz(F.of(e))) bsurv++;
    }
    printf("   n=%-5ld single-bit errors surviving: %ld/%ld ; "
           "single-word bursts surviving: %ld/%ld\n",
           n, survivors, trials, bsurv, trials);
  }
  printf("\n");
}

/* -------------------------------------------------------------- */
static void exp6_rates()
{
  printf("=== 6. Random-rate model: faults in every substep, checks off vs on ===\n");
  Ring R0(R_, S_);
  long ref = reference_smallest(R0, MAXD_);
  double ps[3] = {1e-3, 3e-3, 1e-2};
  for (int pi = 0; pi < 3; pi++) {
    double p = ps[pi];
    long wrongA = 0, wrongB = 0, nA = 0, det = 0, ab = 0;
    for (long t = 0; t < 300; t++) {
      { FaultModel fm(20250806u + (uint64_t)t);
        fm.p_sqr = p; fm.p_mul = p; fm.p_horn = p; fm.p_gcd = p;
        Outcome o = run_once(false, false, 1, &fm);
        if (fm.injected) { nA++; if (o.answer != ref) wrongA++; } }
      { FaultModel fm(20250806u + (uint64_t)t);
        fm.p_sqr = p; fm.p_mul = p; fm.p_horn = p; fm.p_gcd = p;
        Outcome o = run_once(true, true, 1, &fm);
        if (fm.injected) { det += o.detections;
                           if (o.aborted) ab++;
                           else if (o.answer != ref) wrongB++; } }
    }
    printf("   p=%-6g faulty runs %ld/300:  unchecked wrong = %ld,  "
           "checked wrong = %ld  (%ld detections, %ld aborted)\n",
           p, nA, wrongA, wrongB, det, ab);
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  long which = (argc > 1) ? atol(argv[1]) : 0;
  if (!which || which == 1) exp1_identity();
  if (!which || which == 2) exp2_baseline();
  if (!which || which == 3) exp3_sweeps();
  if (!which || which == 4) exp4_window();
  if (!which || which == 5) exp5_foldsize();
  if (!which || which == 6) exp6_rates();
  return 0;
}
