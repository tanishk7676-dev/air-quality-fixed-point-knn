/* main.c - Adaptive Fixed-Point kNN Engine for air-quality classification
 *
 * Usage:  knn_engine [dataset.csv] [--demo]
 *   no flag : interactive menu
 *   --demo  : runs every stage once, non-interactively (good for reports/vivas)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "dataset.h"
#include "normalize.h"
#include "knn.h"
#include "metrics.h"

#define SENSOR_LOG_PATH "data/sensor_log.csv"

static const int K_CANDIDATES[] = {1, 3, 5, 7, 9, 11};
#define N_CAND ((int)(sizeof K_CANDIDATES / sizeof K_CANDIDATES[0]))

/* ---------- program state ---------- */
static Dataset  ds;
static Scaler   sc;
static Prepared pr;
static int      best_k = 3;

/* ---------- helpers ---------- */
static void print_rule(void) { printf("--------------------------------------------\n"); }

/* classify one raw reading and print a friendly report */
static int classify_reading(const float r[NF], int verbose) {
    int16_t q[NF];
    int clamped = 0;
    for (int j = 0; j < NF; j++) {
        if (!scaler_in_range(&sc, r[j], j)) clamped = 1;
        q[j] = scaler_q(&sc, r[j], j);
    }
    int votes = 0;
    int label = knn_predict_fixed(pr.q, pr.label, pr.n, q, best_k, -1, &votes);

    printf("  CO2=%.0f  CO=%.1f  Temp=%.1f  Hum=%.1f  ->  %s  (%d of %d neighbours agree)\n",
           r[0], r[1], r[2], r[3], CLASS_NAMES[label], votes, best_k);
    if (clamped)
        printf("  note: a value is outside the range seen in training - it was clamped. Check the sensor.\n");
    if (verbose && label == NCLASS - 1)
        printf("  *** ALERT: hazardous air - ventilate / leave the area ***\n");
    return label;
}

/* ---------- menu actions ---------- */

/* 1: adaptive k selection, fixed vs float */
static void action_select_k(void) {
    printf("\nAdaptive k selection (leave-one-out, %d samples)\n", pr.n);
    printf(" k | fixed-point %% | float %%\n");
    float acc[N_CAND];
    best_k = knn_select_k(pr.q, pr.label, pr.n, K_CANDIDATES, N_CAND, acc);
    for (int i = 0; i < N_CAND; i++) {
        float fl = knn_loo_float(pr.f, pr.label, pr.n, K_CANDIDATES[i], NULL);
        printf("%2d | %12.2f | %7.2f%s\n", K_CANDIDATES[i], acc[i], fl,
               K_CANDIDATES[i] == best_k ? "   <-- chosen" : "");
    }
    printf("Engine chose k = %d\n", best_k);
}

/* 2: classify typed reading */
static void action_classify_typed(void) {
    char buf[128];
    float r[NF];
    printf("Enter: CO2 CO Temp Humidity   (e.g. 1200 9 31 60)\n> ");
    if (!fgets(buf, sizeof buf, stdin)) return;
    if (sscanf(buf, "%f %f %f %f", &r[0], &r[1], &r[2], &r[3]) != 4) {
        printf("Please enter four numbers separated by spaces.\n");
        return;
    }
    classify_reading(r, 1);
}

/* 3: classify every reading in a sensor log (no labels, 4 numbers per line) */
static void action_classify_log(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { printf("Cannot open %s\n", path); return; }
    char line[128];
    int count[NCLASS] = {0}, total = 0;
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return; }   /* header */
    printf("\nClassifying %s\n", path);
    while (fgets(line, sizeof line, fp)) {
        float r[NF];
        if (sscanf(line, "%f,%f,%f,%f", &r[0], &r[1], &r[2], &r[3]) != 4) continue;
        count[classify_reading(r, 1)]++;
        total++;
    }
    fclose(fp);
    printf("Summary: %d readings", total);
    for (int c = 0; c < NCLASS; c++) printf(" | %s: %d", CLASS_NAMES[c], count[c]);
    printf("\n");
}

/* 4: confusion matrix for fixed-point predictions (leave-one-out) */
static void action_confusion(void) {
    int *pred = malloc((size_t)pr.n * sizeof(int));
    if (!pred) return;
    knn_loo_fixed(pr.q, pr.label, pr.n, best_k, pred);
    Confusion cm;
    conf_init(&cm);
    for (int i = 0; i < pr.n; i++) conf_add(&cm, pr.label[i], pred[i]);
    printf("\nConfusion matrix - fixed-point, k=%d, leave-one-out", best_k);
    conf_print(&cm);
    free(pred);
}

/* 5: float vs fixed-point benchmark */
static void action_benchmark(void) {
    const int REPEATS = 20;
    volatile float sink = 0;
    printf("\nBenchmark: float vs fixed-point (k=%d, %d leave-one-out passes each)\n", best_k, REPEATS);

    clock_t t0 = clock();
    for (int r = 0; r < REPEATS; r++) sink += knn_loo_float(pr.f, pr.label, pr.n, best_k, NULL);
    double ms_float = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;

    t0 = clock();
    for (int r = 0; r < REPEATS; r++) sink += knn_loo_fixed(pr.q, pr.label, pr.n, best_k, NULL);
    double ms_fixed = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    (void)sink;

    int *pf = malloc((size_t)pr.n * sizeof(int)), *pq = malloc((size_t)pr.n * sizeof(int));
    if (!pf || !pq) { free(pf); free(pq); return; }
    float acc_f = knn_loo_float(pr.f, pr.label, pr.n, best_k, pf);
    float acc_q = knn_loo_fixed(pr.q, pr.label, pr.n, best_k, pq);
    int differ = 0;
    for (int i = 0; i < pr.n; i++) if (pf[i] != pq[i]) differ++;
    free(pf); free(pq);

    size_t bytes_f = (size_t)pr.n * NF * sizeof(float);
    size_t bytes_q = (size_t)pr.n * NF * sizeof(int16_t);

    printf("%-26s %12s %12s\n", "", "float", "fixed (Q8)");
    printf("%-26s %11.2f%% %11.2f%%\n", "accuracy", acc_f, acc_q);
    printf("%-26s %10.1f ms %10.1f ms\n", "time (this PC)", ms_float, ms_fixed);
    printf("%-26s %9zu B %11zu B\n", "feature memory", bytes_f, bytes_q);
    printf("%-26s %12d\n", "predictions that differ", differ);
    printf("Memory saved: %.0f%%.  On a PC with a hardware FPU, speed is similar;\n"
           "the speed benefit is expected on microcontrollers without an FPU.\n",
           100.0 * (1.0 - (double)bytes_q / (double)bytes_f));
}

/* 6: why normalisation matters (float kNN, raw vs scaled) */
static void action_norm_experiment(void) {
    printf("\nWhy normalisation matters (float kNN, k=%d, leave-one-out)\n", best_k);
    float raw_acc  = knn_loo_float(ds.raw, ds.label, ds.n, best_k, NULL);
    float norm_acc = knn_loo_float(pr.f,  pr.label, pr.n, best_k, NULL);
    printf("  raw features (CO2 dominates the distance): %6.2f%%\n", raw_acc);
    printf("  normalised features (all fairly weighted): %6.2f%%\n", norm_acc);
}

/* 7: honest hold-out test - k chosen on training part only, scored on unseen part */
static void action_holdout(void) {
    Dataset tr, te;
    if (dataset_split(&ds, &tr, &te, 3) != 0) { printf("out of memory\n"); return; }
    Scaler s2;
    Prepared ptr;
    scaler_fit(&s2, &tr);                                   /* fit on TRAIN only: no leakage */
    if (prepared_build(&ptr, &tr, &s2) != 0) { dataset_free(&tr); dataset_free(&te); return; }

    int k = knn_select_k(ptr.q, ptr.label, ptr.n, K_CANDIDATES, N_CAND, NULL);
    Confusion cm;
    conf_init(&cm);
    for (int i = 0; i < te.n; i++) {
        int16_t q[NF];
        for (int j = 0; j < NF; j++) q[j] = scaler_q(&s2, te.raw[i * NF + j], j);
        conf_add(&cm, te.label[i], knn_predict_fixed(ptr.q, ptr.label, ptr.n, q, k, -1, NULL));
    }
    printf("\nHold-out test: %d train / %d unseen test samples, k chosen on train = %d",
           tr.n, te.n, k);
    conf_print(&cm);

    prepared_free(&ptr);
    dataset_free(&tr);
    dataset_free(&te);
}

/* ---------- start-up / menu ---------- */
static void print_summary(void) {
    int cnt[NCLASS] = {0};
    for (int i = 0; i < ds.n; i++) cnt[ds.label[i]]++;
    printf("Loaded %d readings", ds.n);
    for (int c = 0; c < NCLASS; c++) printf(" | %s: %d", CLASS_NAMES[c], cnt[c]);
    if (ds.skipped) printf(" | %d line(s) skipped", ds.skipped);
    printf("\n");
}

static void run_demo(void) {
    print_rule(); action_select_k();
    print_rule(); action_holdout();
    print_rule(); action_confusion();
    print_rule(); action_benchmark();
    print_rule(); action_norm_experiment();
    print_rule();
    printf("\nClassifying sample readings:\n");
    const float samples[][NF] = {
        {450, 2, 24, 50}, {1200, 9, 31, 60}, {2300, 22, 33, 70}, {65535, 5, 25, 50}
    };
    for (int i = 0; i < 4; i++) classify_reading(samples[i], 1);
    print_rule(); action_classify_log(SENSOR_LOG_PATH);
}

static void menu(void) {
    char buf[32];
    for (;;) {
        printf("\n=== Air Quality kNN Engine (k=%d) ===\n"
               "1) Auto-select k       5) Benchmark float vs fixed\n"
               "2) Classify a reading  6) Why normalise?\n"
               "3) Classify sensor log 7) Hold-out test (unseen data)\n"
               "4) Confusion matrix    0) Exit\n> ", best_k);
        if (!fgets(buf, sizeof buf, stdin)) return;
        switch (atoi(buf)) {
            case 1: action_select_k(); break;
            case 2: action_classify_typed(); break;
            case 3: action_classify_log(SENSOR_LOG_PATH); break;
            case 4: action_confusion(); break;
            case 5: action_benchmark(); break;
            case 6: action_norm_experiment(); break;
            case 7: action_holdout(); break;
            case 0: return;
            default: printf("Choose 0-7.\n");
        }
    }
}

int main(int argc, char **argv) {
    const char *path = "data/air_quality.csv";
    int demo = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--demo") == 0) demo = 1;
        else path = argv[i];
    }

    int n = dataset_load(&ds, path);
    if (n < 0) { fprintf(stderr, "error: cannot open '%s'\n", path); return 1; }
    if (n < 2) { fprintf(stderr, "error: need at least 2 valid rows (got %d)\n", n); dataset_free(&ds); return 1; }

    scaler_fit(&sc, &ds);
    if (prepared_build(&pr, &ds, &sc) != 0) { fprintf(stderr, "error: out of memory\n"); dataset_free(&ds); return 1; }
    print_summary();
    best_k = knn_select_k(pr.q, pr.label, pr.n, K_CANDIDATES, N_CAND, NULL);

    if (demo) run_demo(); else menu();

    prepared_free(&pr);
    dataset_free(&ds);
    return 0;
}
