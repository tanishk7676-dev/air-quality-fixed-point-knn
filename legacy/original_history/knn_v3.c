/* knn_v3.c - fixes found by probe_v2 + sanitizers:
 *   - normalisation clamps out-of-range readings (was overflow / negative)
 *   - zero-range feature (dead sensor) no longer divides by zero
 *   - k is bounded by MAX_K and unfilled neighbour slots are ignored
 *   - one shared normalise function (was duplicated in main)
 *   - ties broken by the nearest neighbour (defensive)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAX_ROWS 1000
#define MAX_K    15
#define NF       4
#define SCALE    256
#define NCLASS   3

typedef struct {
    float   f[NF];
    int16_t x[NF];
    int     label;
} Sample;

static Sample data[MAX_ROWS];
static int    n_rows = 0;
static float  raw[MAX_ROWS][NF];

static float   fmin_[NF], fmax_[NF];
static int32_t qmin_[NF], qmax_[NF];

static int label_from_string(const char *s) {
    if (strcmp(s, "Clean") == 0)     return 0;
    if (strcmp(s, "Moderate") == 0)  return 1;
    return 2;
}

static int load_csv(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[128];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return -1; }
    n_rows = 0;
    while (fgets(line, sizeof line, fp) && n_rows < MAX_ROWS) {
        char lab[16];
        if (sscanf(line, "%f,%f,%f,%f,%15s",
                   &raw[n_rows][0], &raw[n_rows][1], &raw[n_rows][2], &raw[n_rows][3], lab) != 5)
            continue;                              /* skip malformed line */
        data[n_rows].label = label_from_string(lab);
        n_rows++;
    }
    fclose(fp);
    return n_rows;
}

/* ---- shared normalisation (used for training data AND new readings) ---- */
static int16_t normalise_q(float v, int j) {
    if (v < fmin_[j]) v = fmin_[j];                /* clamp to training range */
    if (v > fmax_[j]) v = fmax_[j];
    int32_t range = qmax_[j] - qmin_[j];
    if (range <= 0) return 0;                      /* dead / constant sensor */
    int32_t q = (int32_t)(v * SCALE);
    return (int16_t)(((q - qmin_[j]) * SCALE) / range);
}

static float normalise_f(float v, int j) {
    if (v < fmin_[j]) v = fmin_[j];
    if (v > fmax_[j]) v = fmax_[j];
    float range = fmax_[j] - fmin_[j];
    if (range <= 0.0f) return 0.0f;
    return (v - fmin_[j]) / range;
}

static void normalise_all(void) {
    for (int j = 0; j < NF; j++) {
        fmin_[j] = fmax_[j] = raw[0][j];
        for (int i = 1; i < n_rows; i++) {
            if (raw[i][j] < fmin_[j]) fmin_[j] = raw[i][j];
            if (raw[i][j] > fmax_[j]) fmax_[j] = raw[i][j];
        }
        qmin_[j] = (int32_t)(fmin_[j] * SCALE);
        qmax_[j] = (int32_t)(fmax_[j] * SCALE);
    }
    for (int i = 0; i < n_rows; i++)
        for (int j = 0; j < NF; j++) {
            data[i].f[j] = normalise_f(raw[i][j], j);
            data[i].x[j] = normalise_q(raw[i][j], j);
        }
}

static int32_t dist_fixed(const int16_t *a, const int16_t *b) {
    int32_t s = 0;
    for (int j = 0; j < NF; j++) { int32_t d = a[j] - b[j]; s += d * d; }
    return s;
}

static float dist_float(const float *a, const float *b) {
    float s = 0;
    for (int j = 0; j < NF; j++) { float d = a[j] - b[j]; s += d * d; }
    return s;
}

/* majority vote over the (sorted, nearest-first) neighbour labels; ties -> nearest wins */
static int vote(const int *lab, int k) {
    int votes[NCLASS] = {0}, top = 0;
    for (int i = 0; i < k; i++) if (lab[i] >= 0) votes[lab[i]]++;
    for (int c = 0; c < NCLASS; c++) if (votes[c] > top) top = votes[c];
    for (int i = 0; i < k; i++)
        if (lab[i] >= 0 && votes[lab[i]] == top) return lab[i];
    return 0;
}

static int classify_fixed(const int16_t *q, int k, int skip) {
    if (k < 1) k = 1;
    if (k > MAX_K) k = MAX_K;
    int32_t best[MAX_K]; int lab[MAX_K];
    for (int i = 0; i < k; i++) { best[i] = INT32_MAX; lab[i] = -1; }
    for (int i = 0; i < n_rows; i++) {
        if (i == skip) continue;
        int32_t d = dist_fixed(q, data[i].x);
        int p = k - 1;
        if (d < best[p]) {
            while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; lab[p] = lab[p - 1]; p--; }
            best[p] = d; lab[p] = data[i].label;
        }
    }
    return vote(lab, k);
}

static int classify_float(const float *q, int k, int skip) {
    if (k < 1) k = 1;
    if (k > MAX_K) k = MAX_K;
    float best[MAX_K]; int lab[MAX_K];
    for (int i = 0; i < k; i++) { best[i] = 1e30f; lab[i] = -1; }
    for (int i = 0; i < n_rows; i++) {
        if (i == skip) continue;
        float d = dist_float(q, data[i].f);
        int p = k - 1;
        if (d < best[p]) {
            while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; lab[p] = lab[p - 1]; p--; }
            best[p] = d; lab[p] = data[i].label;
        }
    }
    return vote(lab, k);
}

static float loo_fixed(int k) {
    int ok = 0;
    for (int i = 0; i < n_rows; i++)
        if (classify_fixed(data[i].x, k, i) == data[i].label) ok++;
    return 100.0f * ok / n_rows;
}

static float loo_float(int k) {
    int ok = 0;
    for (int i = 0; i < n_rows; i++)
        if (classify_float(data[i].f, k, i) == data[i].label) ok++;
    return 100.0f * ok / n_rows;
}

int main(void) {
    if (load_csv("data/air_quality.csv") <= 0) { printf("cannot read csv\n"); return 1; }
    normalise_all();
    printf("Loaded %d rows\n\n", n_rows);

    int ks[] = {1, 3, 5, 7};
    int best_k = 1; float best_acc = -1;
    printf(" k | fixed %% | float %%\n");
    for (int i = 0; i < 4; i++) {
        float a = loo_fixed(ks[i]), b = loo_float(ks[i]);
        printf("%2d | %6.2f  | %6.2f\n", ks[i], a, b);
        if (a > best_acc) { best_acc = a; best_k = ks[i]; }
    }
    printf("\nAdaptive choice: k = %d (accuracy %.2f%%)\n", best_k, best_acc);

    float nr[NF] = {1200, 9, 31, 60};
    int16_t q[NF];
    for (int j = 0; j < NF; j++) q[j] = normalise_q(nr[j], j);
    const char *names[] = {"Clean", "Moderate", "Hazardous"};
    printf("New reading -> %s\n", names[classify_fixed(q, best_k, -1)]);
    return 0;
}
