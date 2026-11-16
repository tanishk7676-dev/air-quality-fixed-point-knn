/* dataset.h - loading labelled sensor readings from CSV (dynamic memory) */
#ifndef DATASET_H
#define DATASET_H

#include "config.h"

typedef struct {
    int    n;          /* number of rows loaded                */
    int    cap;        /* allocated capacity (rows)            */
    int    skipped;    /* malformed / unknown-label lines      */
    float *raw;        /* n * NF raw sensor values             */
    int   *label;      /* n class labels (0..NCLASS-1)         */
} Dataset;

int dataset_parse_sensor_row(const char *row, float out[NF]);
int dataset_parse_training_row(const char *row, float out[NF], int *label);

/* returns number of rows loaded, or -1 if the file cannot be opened */
int  dataset_load(Dataset *ds, const char *path);
void dataset_free(Dataset *ds);

/* rows with (index % 10) < test_per_10 go to test, the rest to train */
int  dataset_split(const Dataset *ds, Dataset *train, Dataset *test, int test_per_10);

int dataset_split_provenance(const Dataset *ds, const char *path, Dataset *train, Dataset *test, int by_station);

/* case-insensitive; returns -1 if the text is not a known class */
int  label_from_string(const char *s);

#endif
