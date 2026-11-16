/* metrics.h - confusion matrix, precision, recall */
#ifndef METRICS_H
#define METRICS_H

#include "config.h"

typedef struct {
    int m[NCLASS][NCLASS];     /* m[actual][predicted] */
    int total;
} Confusion;

void  conf_init(Confusion *c);
void  conf_add(Confusion *c, int actual, int predicted);
float conf_accuracy(const Confusion *c);          /* % */
void  conf_print(const Confusion *c);             /* matrix + per-class precision/recall */

#endif
