// Cross-check sieve output against GF2X factor.cpp reference data.
//
//   ./check_ref --ref <reference.txt> [--smax <S>] [--dmin 30] [--dmax 34]
//               [--cert <f.found.cert>]...   (repeatable; per-degree certs)
//               [--survivors <f.survivors.txt>]
//               [--found <f.found.bin>]      (legacy binary image)
//
// Checks performed:
//   1. FALSE NEGATIVES  - a reference s that the sieve calls a survivor.
//                         This is the failure mode check-ntl cannot detect,
//                         since it only validates factors that WERE found.
//   2. DEGREE AGREEMENT - for reference s, the sieve's minimal degree matches.
//   3. POLYNOMIAL AGREEMENT - and the mask matches (factor.cpp reports the
//                         lexicographically least among equal-degree factors,
//                         which for equal-length lowercase hex is the same as
//                         the numerically least, i.e. what the sieve picks).
//   4. FALSE POSITIVES  - sieve reports a degree in [dmin,dmax] for an s the
//                         reference does not list.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <map>
typedef uint64_t u64;

static const u64 SURV = ~0ULL;      // survivor
static const u64 UNK  = 0ULL;       // resolved at a degree we weren't given

int main(int argc, char **argv) {
    const char *refp = nullptr, *survp = nullptr, *foundp = nullptr;
    std::vector<const char *> certs;
    u64 smax = 0; int dmin = 30, dmax = 34;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--ref") && a + 1 < argc) refp = argv[++a];
        else if (!strcmp(argv[a], "--cert") && a + 1 < argc) certs.push_back(argv[++a]);
        else if (!strcmp(argv[a], "--survivors") && a + 1 < argc) survp = argv[++a];
        else if (!strcmp(argv[a], "--found") && a + 1 < argc) foundp = argv[++a];
        else if (!strcmp(argv[a], "--smax") && a + 1 < argc) smax = strtoull(argv[++a], 0, 10);
        else if (!strcmp(argv[a], "--dmin") && a + 1 < argc) dmin = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--dmax") && a + 1 < argc) dmax = atoi(argv[++a]);
        else { fprintf(stderr, "unknown argument '%s'\n", argv[a]); return 1; }
    }
    if (!refp || (certs.empty() && !foundp)) {
        fprintf(stderr,
            "usage: %s --ref <reference.txt> [--smax S] [--dmin D] [--dmax D]\n"
            "          [--cert <f.found.cert>]... [--survivors <f.survivors.txt>]\n"
            "          [--found <f.found.bin>]\n"
            "  Give the cert(s) covering degrees [dmin,dmax] plus the survivors\n"
            "  file from the deepest run, e.g.:\n"
            "     %s --ref d30-34.txt --dmin 30 --dmax 34 \\\n"
            "        --cert out33.found.cert --cert out34.found.cert \\\n"
            "        --survivors out34.survivors.txt\n", argv[0], argv[0]);
        return 1;
    }

    // ---- load reference first (also fixes smax if not supplied) ----
    std::map<u64, std::pair<int,u64>> ref;
    FILE *rf = fopen(refp, "r");
    if (!rf) { perror(refp); return 1; }
    char line[256]; u64 nref = 0, refmax_s = 0;
    while (fgets(line, sizeof line, rf)) {
        u64 s; int d; char p[128];
        if (sscanf(line, "%llu %d p%127s", (unsigned long long*)&s, &d, p) != 3) continue;
        if (d < dmin || d > dmax) continue;
        ref[s] = { d, strtoull(p, 0, 16) };
        nref++; if (s > refmax_s) refmax_s = s;
    }
    fclose(rf);
    if (!smax) smax = refmax_s;

    std::vector<u64> best(smax + 1, UNK);

    // ---- survivors ----
    u64 nsurv = 0;
    if (survp) {
        FILE *sf = fopen(survp, "r");
        if (!sf) { perror(survp); return 1; }
        while (fgets(line, sizeof line, sf)) {
            char *end = nullptr; u64 s = strtoull(line, &end, 10);
            if (end == line || s < 1 || s > smax) continue;
            best[s] = SURV; nsurv++;
        }
        fclose(sf);
    }
    // ---- certs ----
    u64 ncert = 0;
    for (const char *cp : certs) {
        FILE *cf = fopen(cp, "r");
        if (!cf) { perror(cp); return 1; }
        while (fgets(line, sizeof line, cf)) {
            u64 s; int d; char p[128];
            if (sscanf(line, "%llu %d p%127s", (unsigned long long*)&s, &d, p) != 3) continue;
            if (s < 1 || s > smax) continue;
            u64 key = ((u64)d << 48) | strtoull(p, 0, 16);
            if (best[s] == SURV) {
                printf("INCONSISTENT INPUT: s=%llu is in %s but also in the "
                       "survivors file\n", (unsigned long long)s, cp);
            }
            if (best[s] == UNK || key < best[s]) best[s] = key;
            ncert++;
        }
        fclose(cf);
    }
    // ---- legacy binary image ----
    if (foundp) {
        FILE *f = fopen(foundp, "rb");
        if (!f) { perror(foundp); return 1; }
        std::vector<u64> img(smax + 1, SURV);
        if (fread(img.data() + 1, sizeof(u64), smax, f) != smax) {
            fprintf(stderr, "%s: short read (is --smax right?)\n", foundp); return 1; }
        fclose(f);
        for (u64 s = 1; s <= smax; s++) {
            if (img[s] == SURV) { if (best[s] == UNK) best[s] = SURV; }
            else if (img[s] != 0 && (best[s] == UNK || img[s] < best[s])) best[s] = img[s];
        }
    }

    u64 false_neg = 0, deg_mismatch = 0, mask_mismatch = 0, smaller = 0;
    u64 agree = 0, unknown = 0;
    for (auto &kv : ref) {
        u64 s = kv.first; int d = kv.second.first; u64 mask = kv.second.second;
        if (s > smax) continue;
        u64 b = best[s];
        if (b == SURV) {
            if (++false_neg <= 10)
                printf("FALSE NEGATIVE: s=%llu is a survivor but reference has "
                       "degree %d\n", (unsigned long long)s, d);
            continue;
        }
        if (b == UNK) { unknown++; continue; }
        int bd = (int)(b >> 48); u64 bm = b & 0xFFFFFFFFFFFFULL;
        if (bd < d) { smaller++; continue; }
        if (bd != d) { if (++deg_mismatch <= 10)
                printf("DEGREE MISMATCH: s=%llu sieve=%d reference=%d\n",
                       (unsigned long long)s, bd, d); continue; }
        if (bm != mask) { if (++mask_mismatch <= 10)
                printf("POLYNOMIAL MISMATCH: s=%llu d=%d sieve=0x%llx ref=0x%llx\n",
                       (unsigned long long)s, d, (unsigned long long)bm,
                       (unsigned long long)mask); continue; }
        agree++;
    }
    u64 false_pos = 0, hi = (refmax_s < smax ? refmax_s : smax);
    for (u64 s = 1; s <= hi; s++) {
        u64 b = best[s];
        if (b == SURV || b == UNK) continue;
        int bd = (int)(b >> 48);
        if (bd < dmin || bd > dmax) continue;
        if (ref.find(s) == ref.end())
            if (++false_pos <= 10)
                printf("NOT IN REFERENCE: s=%llu sieve reports degree %d\n",
                       (unsigned long long)s, bd);
    }

    printf("\nreference lines in [%d,%d]: %llu  (s up to %llu)\n",
           dmin, dmax, (unsigned long long)nref, (unsigned long long)refmax_s);
    printf("sieve input: %llu cert lines, %llu survivors; checked s = 1..%llu\n",
           (unsigned long long)ncert, (unsigned long long)nsurv,
           (unsigned long long)hi);
    printf("  exact agreement (degree + polynomial): %llu\n", (unsigned long long)agree);
    printf("  sieve found a SMALLER degree (below ref range): %llu\n", (unsigned long long)smaller);
    printf("  reference s with no data supplied (add more certs): %llu\n", (unsigned long long)unknown);
    printf("  FALSE NEGATIVES (ref has factor, sieve says survivor): %llu\n", (unsigned long long)false_neg);
    printf("  degree mismatches: %llu\n", (unsigned long long)deg_mismatch);
    printf("  polynomial mismatches: %llu\n", (unsigned long long)mask_mismatch);
    printf("  sieve in-range but absent from reference: %llu\n", (unsigned long long)false_pos);
    bool bad = false_neg || deg_mismatch || mask_mismatch || false_pos;
    printf("\n%s\n", bad ? "*** DISCREPANCIES FOUND ***" : "ALL CHECKS PASSED");
    return bad ? 1 : 0;
}
