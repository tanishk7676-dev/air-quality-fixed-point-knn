/* knn_v1.c - FIRST DRAFT (single file)
 * Adaptive fixed-point kNN for air-quality classification.
 * Fixed-point format: Q8  (value * 256)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAX_ROWS 1000
#define NF       4
#define SCALE    256
#define NCLASS   3

typedef struct {
    float   f[NF];     /* normalised float features (for comparison) */
    int16_t x[NF];     /* normalised fixed-point features, 0..256     */
    int     label;     /* 0 Clean, 1 Moderate, 2 Hazardous            */
} Sample;

static Sample data[MAX_ROWS];
static int    n_rows = 0;

static float   fmin_[NF], fmax_[NF];
static int32_t qmin_[NF], qmax_[NF];

static int label_from_string(const char *s) {
    if (strcmp(s, "Clean") == 0)     return 0;
    if (strcmp(s, "Moderate") == 0)  return 1;
    return 2;
}

/* raw values are kept temporarily so we can normalise after finding min/max */
static float raw[MAX_ROWS][NF];

static int load_csv(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[128];
    fgets(line, sizeof line, fp);                 /* skip header */
    while (fgets(line, sizeof line, fp) && n_rows < MAX_ROWS) {
        char lab[16];
        sscanf(line, "%f,%f,%f,%f,%15s",
               &raw[n_rows][0], &raw[n_rows][1], &raw[n_rows][2], &raw[n_rows][3], lab);
        data[n_rows].label = label_from_string(lab);
        n_rows++;
    }
    fclose(fp);
    return n_rows;
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
            data[i].f[j] = (raw[i][j] - fmin_[j]) / (fmax_[j] - fmin_[j]);
            int32_t q = (int32_t)(raw[i][j] * SCALE);
            data[i].x[j] = (int16_t)(((q - qmin_[j]) * SCALE) / (qmax_[j] - qmin_[j]));
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

/* classify a fixed-point query against the whole training set */
static int classify_fixed(const int16_t *q, int k) {
    int32_t best[16]; int lab[16];
    for (int i = 0; i < k; i++) best[i] = INT32_MAX;
    for (int i = 0; i < n_rows; i++) {
        int32_t d = dist_fixed(q, data[i].x);
        int p = k - 1;
        if (d < best[p]) {                         /* insertion into sorted list */
            while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; lab[p] = lab[p - 1]; p--; }
            best[p] = d; lab[p] = data[i].label;
        }
    }
    int votes[NCLASS] = {0};
    for (int i = 0; i < k; i++) votes[lab[i]]++;
    int win = 0;
    for (int c = 1; c < NCLASS; c++) if (votes[c] > votes[win]) win = c;
    return win;
}

static int classify_float(const float *q, int k) {
    float best[16]; int lab[16];
    for (int i = 0; i < k; i++) best[i] = 1e30f;
    for (int i = 0; i < n_rows; i++) {
        float d = dist_float(q, data[i].f);
        int p = k - 1;
        if (d < best[p]) {
            while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; lab[p] = lab[p - 1]; p--; }
            best[p] = d; lab[p] = data[i].label;
        }
    }
    int votes[NCLASS] = {0};
    for (int i = 0; i < k; i++) votes[lab[i]]++;
    int win = 0;
    for (int c = 1; c < NCLASS; c++) if (votes[c] > votes[win]) win = c;
    return win;
}

/* leave-one-out accuracy: classify every training sample using the training set */
static float loo_fixed(int k) {
    int ok = 0;
    for (int i = 0; i < n_rows; i++)
        if (classify_fixed(data[i].x, k) == data[i].label) ok++;
    return 100.0f * ok / n_rows;
}

static float loo_float(int k) {
    int ok = 0;
    for (int i = 0; i < n_rows; i++)
        if (classify_float(data[i].f, k) == data[i].label) ok++;
    return 100.0f * ok / n_rows;
}

int main(void) {
    if (load_csv("data/air_quality.csv") < 0) { printf("cannot open csv\n"); return 1; }
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

    /* classify one new reading: CO2=1200, CO=9, Temp=31, Hum=60 */
    float nr[NF] = {1200, 9, 31, 60};
    int16_t q[NF];
    for (int j = 0; j < NF; j++) {
        int32_t v = (int32_t)(nr[j] * SCALE);
        q[j] = (int16_t)(((v - qmin_[j]) * SCALE) / (qmax_[j] - qmin_[j]));
    }
    const char *names[] = {"Clean", "Moderate", "Hazardous"};
    printf("New reading -> %s\n", names[classify_fixed(q, best_k)]);
    return 0;
}
