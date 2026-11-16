/* probe_v2.c - edge-case probe for knn_v2.c (includes it directly). */
#define main main_v2
#include "knn_v2.c"
#undef main

static int norm_q(float v, int j) {
    int32_t q = (int32_t)(v * SCALE);
    return (int16_t)(((q - qmin_[j]) * SCALE) / (qmax_[j] - qmin_[j]));
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    load_csv("data/air_quality.csv");
    normalise_all();

    printf("[1] Glitchy sensor value: CO2 = 65535 ppm\n");
    printf("    normalised -> %d (expected: clamp to 256)\n", norm_q(65535.0f, 0));

    printf("[2] Reading below range: CO2 = 100 ppm\n");
    printf("    normalised -> %d (expected: clamp to 0)\n", norm_q(100.0f, 0));

    printf("[3] Vote ties in leave-one-out, k=3 and k=5\n");
    int ks[2] = {3, 5};
    for (int t = 0; t < 2; t++) {
        int k = ks[t], ties = 0;
        for (int i = 0; i < n_rows; i++) {
            int32_t best[16]; int lab[16];
            for (int a = 0; a < k; a++) best[a] = INT32_MAX;
            for (int r = 0; r < n_rows; r++) {
                if (r == i) continue;
                int32_t d = dist_fixed(data[i].x, data[r].x);
                int p = k - 1;
                if (d < best[p]) { while (p > 0 && best[p-1] > d) { best[p]=best[p-1]; lab[p]=lab[p-1]; p--; } best[p]=d; lab[p]=data[r].label; }
            }
            int votes[NCLASS] = {0};
            for (int a = 0; a < k; a++) votes[lab[a]]++;
            int top = 0; for (int c = 0; c < NCLASS; c++) if (votes[c] > top) top = votes[c];
            int cnt = 0; for (int c = 0; c < NCLASS; c++) if (votes[c] == top) cnt++;
            if (cnt > 1) ties++;
        }
        printf("    k=%d: %d of %d samples had a tied vote\n", k, ties, n_rows);
    }

    fflush(stdout);
    printf("[4] k larger than the internal buffer (k = 20)\n");
    printf("    result = %d\n", classify_fixed(data[0].x, 20, 0));
    return 0;
}
