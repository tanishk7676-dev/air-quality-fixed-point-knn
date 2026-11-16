/* test_knn.c - unit + regression tests. Run from the project root: make test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "dataset.h"
#include "normalize.h"
#include "knn.h"
#include "metrics.h"

static int passed = 0, failed = 0;
#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  PASS  %s\n", name); } \
    else      { failed++; printf("  FAIL  %s\n", name); } } while (0)

/* build a tiny Prepared set by hand: n points, NF features each (0..256 scale) */
static void tiny_set(Prepared *p, int n, const int16_t *q, const int *lab) {
    p->n = n;
    p->q = malloc((size_t)n * NF * sizeof(int16_t));
    p->f = malloc((size_t)n * NF * sizeof(float));
    p->label = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        p->label[i] = lab[i];
        for (int j = 0; j < NF; j++) {
            p->q[i * NF + j] = q[i * NF + j];
            p->f[i * NF + j] = q[i * NF + j] / 256.0f;
        }
    }
}

static void write_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "w");
    if (fp) { fputs(text, fp); fclose(fp); }
}

int main(void) {
    printf("Label parsing (bug: v1-v3 turned any unknown text into 'Hazardous')\n");
    CHECK("'Clean' -> 0",              label_from_string("Clean") == 0);
    CHECK("'clean' lowercase -> 0",    label_from_string("clean") == 0);
    CHECK("'HAZARDOUS' uppercase -> 2", label_from_string("HAZARDOUS") == 2);
    CHECK("typo 'Hazardus' -> -1",     label_from_string("Hazardus") == -1);
    CHECK("garbage 'banana' -> -1",    label_from_string("banana") == -1);

    printf("\nCSV loading\n");
    write_file("knn_test_tmp.csv",
               "co2,co,temp,humidity,label\n"
               "400,1.0,20.0,40.0,Clean\n"
               "1000,6.0,25.0,50.0,Moderate\n"
               "oops,this,is,bad\n"
               "2000,20.0,30.0,60.0,Hazardus\n"       /* typo -> skipped */
               "\n"
               "2200,22.0,31.0,65.0,Hazardous\r\n");   /* Windows line ending */
    Dataset ds;
    int n = dataset_load(&ds, "knn_test_tmp.csv");
    CHECK("loads 3 valid rows",            n == 3);
    CHECK("counts 2 skipped lines",        ds.skipped == 2);
    CHECK("handles CRLF on last row",      ds.n == 3 && ds.label[2] == 2);
    CHECK("missing file returns -1",       dataset_load(&(Dataset){0}, "/tmp/does_not_exist.csv") == -1);

    printf("\nNormalisation (bugs: overflow on glitch value, divide-by-zero)\n");
    Scaler sc;
    scaler_fit(&sc, &ds);
    CHECK("min maps to 0",                 scaler_q(&sc, 400.0f, 0) == 0);
    CHECK("max maps to 256",               scaler_q(&sc, 2200.0f, 0) == 256);
    CHECK("mid value is about 128",        scaler_q(&sc, 1300.0f, 0) >= 127 && scaler_q(&sc, 1300.0f, 0) <= 129);
    CHECK("65535 clamps to 256",           scaler_q(&sc, 65535.0f, 0) == 256);
    CHECK("below range clamps to 0",       scaler_q(&sc, -50.0f, 0) == 0);
    CHECK("absurd 1e30 clamps to 256",     scaler_q(&sc, 1e30f, 0) == 256);
    CHECK("in_range true inside",          scaler_in_range(&sc, 1000.0f, 0) == 1);
    CHECK("in_range false outside",        scaler_in_range(&sc, 65535.0f, 0) == 0);
    dataset_free(&ds);
    remove("knn_test_tmp.csv");

    Dataset flat;                                         /* dead sensor: all CO = 5 */
    memset(&flat, 0, sizeof flat);
    flat.n = 2; flat.raw = malloc(2 * NF * sizeof(float)); flat.label = malloc(2 * sizeof(int));
    float fr[2 * NF] = {400, 5, 20, 40,   900, 5, 22, 45};
    memcpy(flat.raw, fr, sizeof fr); flat.label[0] = 0; flat.label[1] = 1;
    Scaler fs;
    scaler_fit(&fs, &flat);
    CHECK("constant column maps to 0 (no crash)", scaler_q(&fs, 5.0f, 1) == 0 && scaler_f(&fs, 5.0f, 1) == 0.0f);
    dataset_free(&flat);

    printf("\nkNN core (bugs: leave-one-out self-match, k overflow, ties)\n");
    /* two well separated clusters on feature 0 */
    const int16_t pts[6 * NF] = {
        10,0,0,0,   20,0,0,0,   15,0,0,0,        /* class 0 */
        240,0,0,0,  250,0,0,0,  245,0,0,0 };     /* class 2 */
    const int labs[6] = {0, 0, 0, 2, 2, 2};
    Prepared p;
    tiny_set(&p, 6, pts, labs);
    int16_t qa[NF] = {12, 0, 0, 0}, qb[NF] = {248, 0, 0, 0};
    CHECK("predicts class 0 near cluster 0", knn_predict_fixed(p.q, p.label, p.n, qa, 3, -1, NULL) == 0);
    CHECK("predicts class 2 near cluster 2", knn_predict_fixed(p.q, p.label, p.n, qb, 3, -1, NULL) == 2);
    CHECK("float version agrees",            knn_predict_float(p.f, p.label, p.n, &p.f[0], 3, 0, NULL) == 0);
    int votes = 0;
    knn_predict_fixed(p.q, p.label, p.n, qa, 3, -1, &votes);
    CHECK("votes_out reports 3 of 3",        votes == 3);
    CHECK("k=100 is clamped, no overflow",   knn_predict_fixed(p.q, p.label, p.n, qa, 100, -1, NULL) >= 0);
    CHECK("k=0 treated as k=1",              knn_predict_fixed(p.q, p.label, p.n, qa, 0, -1, NULL) == 0);
    CHECK("LOO k=1 on clean clusters = 100%", knn_loo_fixed(p.q, p.label, p.n, 1, NULL) == 100.0f);
    prepared_free(&p);

    /* v1 bug: LOO counted the sample itself. Two points, different labels: each is
     * classified by the OTHER point, so accuracy must be 0%, not 100%. */
    const int16_t two[2 * NF] = {0,0,0,0,  200,0,0,0};
    const int tl[2] = {0, 1};
    tiny_set(&p, 2, two, tl);
    CHECK("LOO excludes the sample itself (0%)", knn_loo_fixed(p.q, p.label, p.n, 1, NULL) == 0.0f);
    prepared_free(&p);

    /* tie: three neighbours, three different labels -> nearest neighbour's class wins */
    const int16_t tie[3 * NF] = {100,0,0,0,  110,0,0,0,  120,0,0,0};
    const int ll[3] = {2, 0, 1};
    tiny_set(&p, 3, tie, ll);
    int16_t qt[NF] = {101, 0, 0, 0};
    CHECK("3-way tie goes to nearest neighbour", knn_predict_fixed(p.q, p.label, p.n, qt, 3, -1, NULL) == 2);
    prepared_free(&p);

    printf("\nAdaptive k + fixed vs float on the real dataset\n");
    Dataset real;
    if (dataset_load(&real, "data/air_quality.csv") > 0) {
        Scaler rs;
        Prepared rp;
        scaler_fit(&rs, &real);
        prepared_build(&rp, &real, &rs);
        const int cands[] = {1, 3, 5, 7, 9, 11, 13, 15};
        float acc[8];
        int k = knn_select_k(rp.q, rp.label, rp.n, cands, 8, acc);
        float best = 0;
        for (int i = 0; i < 8; i++) if (acc[i] > best) best = acc[i];
        CHECK("selected k is a candidate",       k >= 1 && k <= 15 && (k % 2) == 1);
        CHECK("selected k has the best accuracy", knn_loo_fixed(rp.q, rp.label, rp.n, k, NULL) == best);
        CHECK("accuracy is realistic (not ~100%)", best > 80.0f && best < 97.0f);

        int *pf = malloc((size_t)rp.n * sizeof(int)), *pq = malloc((size_t)rp.n * sizeof(int));
        knn_loo_float(rp.f, rp.label, rp.n, k, pf);
        knn_loo_fixed(rp.q, rp.label, rp.n, k, pq);
        int differ = 0;
        for (int i = 0; i < rp.n; i++) if (pf[i] != pq[i]) differ++;
        CHECK("fixed and float differ on < 2% of predictions", differ * 100 < rp.n * 2);
        free(pf); free(pq);
        prepared_free(&rp);
        dataset_free(&real);
    } else {
        CHECK("data/air_quality.csv found (run tests from project root)", 0);
    }

    printf("\nConfusion matrix\n");
    Confusion cm;
    conf_init(&cm);
    conf_add(&cm, 0, 0); conf_add(&cm, 0, 1); conf_add(&cm, 1, 1); conf_add(&cm, 2, 2);
    conf_add(&cm, 5, 0);                                   /* invalid class ignored */
    CHECK("total counts valid entries only", cm.total == 4);
    CHECK("accuracy = 3/4 = 75%",            conf_accuracy(&cm) == 75.0f);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed != 0;
}
