/* metrics.c */
#include "metrics.h"
#include <stdio.h>
#include <string.h>

void conf_init(Confusion *c) { memset(c, 0, sizeof *c); }

void conf_add(Confusion *c, int actual, int predicted) {
    if (actual < 0 || actual >= NCLASS || predicted < 0 || predicted >= NCLASS) return;
    c->m[actual][predicted]++;
    c->total++;
}

float conf_accuracy(const Confusion *c) {
    int ok = 0;
    for (int i = 0; i < NCLASS; i++) ok += c->m[i][i];
    return c->total ? 100.0f * ok / c->total : 0.0f;
}

void conf_print(const Confusion *c) {
    printf("\n%-22s", "actual\\pred");
    for (int j = 0; j < NCLASS; j++) printf("%22s", CLASS_NAMES[j]);
    printf("\n");
    for (int i = 0; i < NCLASS; i++) {
        printf("%-22s", CLASS_NAMES[i]);
        for (int j = 0; j < NCLASS; j++) printf("%22d", c->m[i][j]);
        printf("\n");
    }
    printf("\n%-22s%22s%22s\n", "class", "precision%", "recall%");
    double macro_f1 = 0;
    for (int k = 0; k < NCLASS; k++) {
        int tp = c->m[k][k], col = 0, row = 0;
        for (int i = 0; i < NCLASS; i++) { col += c->m[i][k]; row += c->m[k][i]; }
        float prec = col ? 100.0f * tp / col : 0.0f;
        float rec  = row ? 100.0f * tp / row : 0.0f;
        printf("%-22s%22.1f%22.1f\n", CLASS_NAMES[k], prec, rec);
        if (prec+rec>0) macro_f1 += 2.0*prec*rec/(prec+rec);
    }
    printf("macro-F1: %.2f%%\n", macro_f1/NCLASS);
    printf("overall accuracy: %.2f%%  (%d samples)\n", conf_accuracy(c), c->total);
}
