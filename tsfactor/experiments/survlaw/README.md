# Survival law for other prime degrees (round 11)

Question (Gord): does the empirical law S(k) ~ (16/9)/k -- the fraction of
trinomials x^r + x^s + 1 with no factor of degree <= k -- hold for prime r
that are not Mersenne exponents, i.e. how general must tsfactor's schedule be?

    g++ -O2 -o survlaw survlaw.cpp -lntl -lgf2x -lgmp -pthread
    ./survlaw 3217 4253 4423 9689 9941 11213 19937 21701 23209 > mers_out.txt
    ./survlaw 9001 9433 9011 9419 9013 9421 9007 9431 \
              19001 19417 19051 19483 19013 19421 19031 19447 > nonmers_out.txt
    python3 analyse.py mers_out.txt nonmers_out.txt > results.txt

survlaw finds the least factor degree of every s in [1, (r-1)/2] by
incremental distinct-degree factorisation (NTL; about 7 minutes for all 25
degrees on one core).  results.txt has k*S(k) per r and pooled; the
conclusion is in docs/harness_design.md ("Survival law") at the top level.
