/* probe_v3.c - re-runs the edge cases that broke v2, as PASS/FAIL checks. */
#define main main_v3
#include "knn_v3.c"
#undef main

static int fails = 0;
#define CHECK(name, cond) do { printf("  %-52s %s\n", name, (cond) ? "PASS" : "FAIL"); if (!(cond)) fails++; } while (0)

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    load_csv("data/air_quality.csv");
    normalise_all();

    CHECK("CO2=65535 clamps to 256",          normalise_q(65535.0f, 0) == 256);
    CHECK("CO2=100 (below range) clamps to 0", normalise_q(100.0f, 0) == 0);
    CHECK("CO2=1e30 (absurd value) clamps",    normalise_q(1e30f, 0) == 256);
    CHECK("k=20 does not overflow",            classify_fixed(data[0].x, 20, 0) >= 0);
    CHECK("k=0 treated as k=1",                classify_fixed(data[0].x, 0, 0) >= 0);

    /* dead sensor: force column 1 constant, re-normalise */
    for (int i = 0; i < n_rows; i++) raw[i][1] = 5.0f;
    normalise_all();
    CHECK("constant column: no crash, maps to 0", normalise_q(5.0f, 1) == 0 && data[0].x[1] == 0);

    printf(fails ? "\n%d check(s) FAILED\n" : "\nAll checks passed\n", fails);
    return fails != 0;
}
