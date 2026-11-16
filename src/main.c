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
#include "aqi.h"
#include <math.h>
#include <errno.h>
#include <ctype.h>
#include <limits.h>

#define SENSOR_LOG_PATH "data/sensor_log.csv"

static const int K_CANDIDATES[] = {1, 3, 5, 7, 9, 11, 13, 15};   /* up to MAX_K */
#define N_CAND ((int)(sizeof K_CANDIDATES / sizeof K_CANDIDATES[0]))

/* ---------- program state ---------- */
static Dataset  ds;
static Scaler   sc;
static Prepared pr;
static Prepared pr_full;
#define EVAL_LIMIT 1200
/* Deterministic, evenly spaced sample for quadratic-cost LOOCV. */
static int make_eval_sample(const Dataset *source, Dataset *out) {
    memset(out, 0, sizeof *out);
    int n = source->n < EVAL_LIMIT ? source->n : EVAL_LIMIT;
    out->raw = malloc((size_t)n * NF * sizeof(float));
    out->label = malloc((size_t)n * sizeof(int));
    if (!out->raw || !out->label) { dataset_free(out); return -1; }
    out->n = out->cap = n;
    for (int i = 0; i < n; i++) {
        int index = (int)((long long)i * source->n / n);
        memcpy(out->raw + (size_t)i * NF, source->raw + (size_t)index * NF, NF * sizeof(float));
        out->label[i] = source->label[index];
    }
    return 0;
}

static int      best_k = 3;
/* Read a complete console line, discarding any excess input. */
static int read_line(char *buf, size_t size) {
    if (!fgets(buf, (int)size, stdin)) return 0;
    if (!strchr(buf, '\n') && !feof(stdin)) {
        int ch;
        while ((ch = getchar()) != '\n' && ch != EOF) {}
        return -1;
    }
    return 1;
}
static int valid_reading(float value) {
    /* The calculator supports finite, nonnegative concentrations only. */
    return isfinite(value) && value >= 0.0f && value <= 1000000.0f;
}


/* ---------- helpers ---------- */
static void print_rule(void) { printf("--------------------------------------------\n"); }

/* classify one raw reading and print a friendly report */
static int classify_reading(const float r[NF], int verbose) {
    for (int j=0;j<NF;j++) {
        if (!valid_reading(r[j])) { puts("  Invalid or unsupported pollutant concentration."); return -1; }
    }
    int dominant=-1;
    int aqi=aqi_calculate(r,&dominant);
    if (aqi<0) { puts("  Cannot calculate AQI: need 3 pollutants including PM2.5 or PM10."); return -1; }
    int16_t q[NF]; int clamped=0;
    for(int j=0;j<NF;j++) {
        if (r[j]>=0 && !scaler_in_range(&sc,r[j],j)) clamped=1;
        q[j]=scaler_q(&sc,r[j],j);
    }
    int votes=0;
    int prediction=knn_predict_weighted_fixed(pr_full.q,pr_full.label,pr_full.n,q,best_k,-1,&votes);
    printf("  CPCB AQI: %d (%s), dominant: %s\n",aqi,CLASS_NAMES[aqi_category(aqi)],FEATURE_NAMES[dominant]);
    printf("  kNN predicted (distance-weighted): %s (%d/%d neighbours of winning class)\n",CLASS_NAMES[prediction],votes,best_k);
    if (clamped) puts("  Note: input outside training range; kNN clamps it.");
    if (verbose && aqi>200) puts("  Air quality warning: reduce prolonged outdoor exposure.");
    return aqi_category(aqi);
}

/* ---------- menu actions ---------- */

/* 1: adaptive k selection, fixed vs float */
static void action_select_k(void) {
    printf("\nAdaptive k selection (leave-one-out, %d samples)\n", pr.n);
    printf(" k | weighted fixed %% | float majority %%\n");
    float acc[N_CAND];
    best_k = knn_select_weighted_k(pr.q, pr.label, pr.n, K_CANDIDATES, N_CAND, acc);
    for (int i = 0; i < N_CAND; i++) {
        float fl = knn_loo_float(pr.f, pr.label, pr.n, K_CANDIDATES[i], NULL);
        printf("%2d | %12.2f | %7.2f%s\n", K_CANDIDATES[i], acc[i], fl,
               K_CANDIDATES[i] == best_k ? "   <-- chosen" : "");
    }
    printf("Engine chose k = %d\n", best_k);
}

/* 2: classify typed reading */
static void action_classify_typed(void) {
    char buf[256]; float r[NF];
    puts("Enter seven averaged pollutant concentrations:");
    puts("PM2.5 PM10 NO2 SO2 CO O3 NH3");
    puts("CO in mg/m3, others in ug/m3. Use -1 for missing values.");
    puts("24h averages except CO and O3 (8h averages).");
    printf("> ");
    int input_status = read_line(buf, sizeof buf);
    if (input_status == 0) return;
    if (input_status < 0) { puts("Input line too long."); return; }
    char extra;
    if (sscanf(buf,"%f %f %f %f %f %f %f %c",&r[0],&r[1],&r[2],&r[3],&r[4],&r[5],&r[6],&extra)!=7) {
        puts("Please enter exactly seven numbers.");return;
    }
    for(int j=0;j<NF;j++) if (!isfinite(r[j]) || r[j]<-1 || r[j]>1000000.0f) {puts("Invalid concentration.");return;}
    for(int j=0;j<NF;j++) if(r[j]<0) {puts("kNN needs seven measured pollutants; official AQI can use partial inputs."); int dominant; int a=aqi_calculate(r,&dominant); if(a<0) puts("Insufficient data."); else printf("CPCB AQI: %d (%s)\n",a,CLASS_NAMES[aqi_category(a)]);return;}
    classify_reading(r,1);
}

/* 3: classify every reading in a sensor log (no labels, 7 numbers per line) */
static void action_classify_log(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { printf("Cannot open %s\n", path); return; }
    char line[512];
    int count[NCLASS] = {0}, total = 0;
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return; }   /* header */
    printf("\nClassifying %s\n", path);
    while (fgets(line, sizeof line, fp)) {
        if (!strchr(line, '\n') && !feof(fp)) {
            int ch; while ((ch=fgetc(fp)) != '\n' && ch != EOF) {}
            fprintf(stderr, "Skipping overlong sensor row.\n"); continue;
        }
        float r[NF];
        if (dataset_parse_sensor_row(line, r) != 0) { fprintf(stderr, "Skipping invalid sensor row.\n"); continue; }
        int category=classify_reading(r, 1);
        if(category<0) continue;
        count[category]++;
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
    knn_loo_weighted_fixed(pr.q, pr.label, pr.n, best_k, pred);
    Confusion cm;
    conf_init(&cm);
    for (int i = 0; i < pr.n; i++) conf_add(&cm, pr.label[i], pred[i]);
    printf("\nConfusion matrix - distance-weighted fixed-point, k=%d, leave-one-out", best_k);
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
    printf("Memory saved: %.0f%%.\n"
           "Speed: this PC has a hardware FPU, so fixed-point shows no speed advantage here.\n"
           "The speed benefit applies to microcontrollers without an FPU (not measured in this program).\n",
           100.0 * (1.0 - (double)bytes_q / (double)bytes_f));
}

/* 6: why normalisation matters (float kNN, raw vs scaled) */
static void action_norm_experiment(void) {
    printf("\nWhy normalisation matters (float kNN, k=%d, leave-one-out)\n", best_k);
    /* Reconstruct the same sampled rows in raw units for a controlled comparison. */
    float *raw = malloc((size_t)pr.n * NF * sizeof(float));
    if (!raw) { puts("Out of memory"); return; }
    for (int i = 0; i < pr.n; i++) {
        int index = (int)((long long)i * ds.n / pr.n);
        memcpy(raw + (size_t)i * NF, ds.raw + (size_t)index * NF, NF * sizeof(float));
    }
    float raw_acc = knn_loo_float(raw, pr.label, pr.n, best_k, NULL);
    free(raw);
    float norm_acc = knn_loo_float(pr.f,  pr.label, pr.n, best_k, NULL);
    printf("  raw features (large-unit pollutants dominate): %6.2f%%\n", raw_acc);
    printf("  normalised features (same %d rows): %6.2f%%\n", pr.n, norm_acc);
}

/* 7: honest hold-out test - k chosen on training part only, scored on unseen part */
static void action_holdout_mode(int mode) {
    Dataset tr, te;
    int split_ok = mode == 0 ? dataset_split(&ds, &tr, &te, 3) :
        dataset_split_provenance(&ds, "data/real_training_provenance.csv", &tr, &te, mode == 1);
    if (split_ok != 0) { printf("Cannot split dataset; check provenance alignment or memory.\n"); return; }
    Scaler s2;
    Prepared ptr;
    scaler_fit(&s2, &tr);                                   /* fit on TRAIN only: no leakage */
    if (prepared_build(&ptr, &tr, &s2) != 0) { dataset_free(&tr); dataset_free(&te); return; }

    Dataset eval_tr; Prepared eval_ptr;
    if (make_eval_sample(&tr, &eval_tr) != 0) { prepared_free(&ptr); dataset_free(&tr); dataset_free(&te); return; }
    if (prepared_build(&eval_ptr, &eval_tr, &s2) != 0) { dataset_free(&eval_tr); prepared_free(&ptr); dataset_free(&tr); dataset_free(&te); return; }
    int k = knn_select_weighted_k(eval_ptr.q, eval_ptr.label, eval_ptr.n, K_CANDIDATES, N_CAND, NULL);
    printf("  k chosen using %d training-only samples for LOOCV\n", eval_tr.n);
    prepared_free(&eval_ptr); dataset_free(&eval_tr);
    Confusion cm, baseline;
    conf_init(&cm); conf_init(&baseline);
    for (int i = 0; i < te.n; i++) {
        int16_t q[NF];
        for (int j = 0; j < NF; j++) q[j] = scaler_q(&s2, te.raw[i * NF + j], j);
        conf_add(&cm, te.label[i], knn_predict_weighted_fixed(ptr.q, ptr.label, ptr.n, q, k, -1, NULL));
        conf_add(&baseline, te.label[i], knn_predict_fixed(ptr.q, ptr.label, ptr.n, q, k, -1, NULL));
    }
    printf("\n%s hold-out: %d train / %d test samples, k chosen on train = %d\n",
           mode == 0 ? "Row-interleaved" : mode == 1 ? "Unseen-station" : "Future-date (2020)",
           tr.n, te.n, k);
    printf("Majority-vote baseline on same test set: %.2f%%\n", conf_accuracy(&baseline));
    conf_print(&cm);

    prepared_free(&ptr);
    dataset_free(&tr);
    dataset_free(&te);
}

/* ---------- start-up / menu ---------- */
static void action_holdout(void) { action_holdout_mode(0); }

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
    print_rule(); action_holdout_mode(1);
    print_rule(); action_holdout_mode(2);
    print_rule(); action_confusion();
    print_rule(); action_benchmark();
    print_rule(); action_norm_experiment();
    print_rule();
    printf("\nClassifying sample readings:\n");
    const float samples[][NF] = {
        {20,40,20,15,0.5,30,100},
        {75,180,90,70,2,120,350},
        {150,370,300,900,12,250,1000},
        {280,500,450,1700,40,800,2000}
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
               "4) Confusion matrix    8) Unseen-station test\n"
               "9) Future-date test   0) Exit\n> ", best_k);
        int status = read_line(buf, sizeof buf);
        if (status == 0) return;
        if (status < 0) { puts("Menu input too long. Choose 0-9."); continue; }
        errno = 0;
        char *end;
        long choice = strtol(buf, &end, 10);
        while (isspace((unsigned char)*end)) end++;
        if (errno == ERANGE || end == buf || *end != '\0' || choice < 0 || choice > 9) {
            puts("Invalid menu choice. Enter a whole number from 0 to 9.");
            continue;
        }
        switch ((int)choice) {
            case 1: action_select_k(); break;
            case 2: action_classify_typed(); break;
            case 3: action_classify_log(SENSOR_LOG_PATH); break;
            case 4: action_confusion(); break;
            case 5: action_benchmark(); break;
            case 6: action_norm_experiment(); break;
            case 7: action_holdout(); break;
            case 8: action_holdout_mode(1); break;
            case 9: action_holdout_mode(2); break;
            case 0: return;
            default: printf("Choose 0-9.\n");
        }
    }
}

int main(int argc, char **argv) {
    const char *path = "data/real_training.csv";
    int demo = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--demo") == 0) demo = 1;
        else path = argv[i];
    }

    int n = dataset_load(&ds, path);
    if (n < 0) { fprintf(stderr, "error: cannot open '%s'\n", path); return 1; }
    if (n < 2) { fprintf(stderr, "error: need at least 2 valid rows (got %d)\n", n); dataset_free(&ds); return 1; }

    scaler_fit(&sc, &ds);
    Dataset eval_ds;
    if (make_eval_sample(&ds, &eval_ds) != 0) { dataset_free(&ds); return 1; }
    if (prepared_build(&pr_full, &ds, &sc) != 0) { dataset_free(&eval_ds); dataset_free(&ds); return 1; }
    if (prepared_build(&pr, &eval_ds, &sc) != 0) { fprintf(stderr, "error: out of memory\n"); prepared_free(&pr_full); dataset_free(&eval_ds); dataset_free(&ds); return 1; }
    dataset_free(&eval_ds);
    print_summary();
    printf("LOOCV and benchmarks use %d evenly spaced records; predictions use all %d.\n", pr.n, pr_full.n);
    best_k = knn_select_weighted_k(pr.q, pr.label, pr.n, K_CANDIDATES, N_CAND, NULL);

    if (demo) run_demo(); else menu();

    prepared_free(&pr);
    prepared_free(&pr_full);
    dataset_free(&ds);
    return 0;
}
