/* dataset.c */
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

const char *const CLASS_NAMES[NCLASS]   = {"Good", "Satisfactory", "Moderately polluted", "Poor", "Very Poor", "Severe"};
const char *const FEATURE_NAMES[NF]     = {"PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"};

/* portable case-insensitive compare (strcasecmp is not standard C) */
static int str_ieq(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

int label_from_string(const char *s) {
    for (int c = 0; c < NCLASS; c++)
        if (str_ieq(s, CLASS_NAMES[c])) return c;
    return -1;
}

static int ds_push(Dataset *ds, const float *f, int label) {
    if (ds->n == ds->cap) {
        int ncap = ds->cap ? ds->cap * 2 : 64;
        float *nr = realloc(ds->raw, (size_t)ncap * NF * sizeof(float));
        if (!nr) return -1;
        ds->raw = nr;
        int *nl = realloc(ds->label, (size_t)ncap * sizeof(int));
        if (!nl) return -1;
        ds->label = nl;
        ds->cap = ncap;
    }
    memcpy(&ds->raw[ds->n * NF], f, NF * sizeof(float));
    ds->label[ds->n] = label;
    ds->n++;
    return 0;
}

/* Preserve empty CSV cells and reject extra columns, NaN and negative readings. */
static int split_fields(char *line, char **fields, int expected) {
    int n = 0; char *start = line;
    for (char *p = line; ; p++) {
        if (*p == ',' || *p == '\n' || *p == '\r' || *p == '\0') {
            char end = *p; *p = '\0';
            if (n >= expected) return -1;
            while (isspace((unsigned char)*start)) start++;
            char *tail = start + strlen(start);
            while (tail > start && isspace((unsigned char)tail[-1])) *--tail = '\0';
            fields[n++] = start;
            if (end != ',') break;
            start = p + 1;
        }
    }
    return n == expected ? 0 : -1;
}

int dataset_parse_sensor_row(const char *row, float out[NF]) {
    char buf[2048]; char *fields[NF];
    if (strlen(row) >= sizeof buf) return -1;
    strcpy(buf, row);
    if (split_fields(buf, fields, NF)) return -1;
    for (int j = 0; j < NF; j++) {
        if (!*fields[j]) return -1;
        char *end;
        out[j] = strtof(fields[j], &end);
        if (end == fields[j] || *end || !isfinite(out[j]) || out[j] < 0) return -1;
    }
    return 0;
}

int dataset_parse_training_row(const char *row, float out[NF], int *label) {
    char buf[2048]; char *fields[NF + 1];
    if (strlen(row) >= sizeof buf) return -1;
    strcpy(buf, row);
    if (split_fields(buf, fields, NF + 1)) return -1;
    for (int j = 0; j < NF; j++) {
        if (!*fields[j]) return -1;
        char *end;
        out[j] = strtof(fields[j], &end);
        if (end == fields[j] || *end || !isfinite(out[j]) || out[j] < 0) return -1;
    }
    *label = label_from_string(fields[NF]);
    return *label >= 0 ? 0 : -1;
}

int dataset_load(Dataset *ds, const char *path) {
    memset(ds, 0, sizeof *ds);
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[1024];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return 0; }
    while (fgets(line, sizeof line, fp)) {
        if (!strchr(line, '\n') && !feof(fp)) {
            int ch; while ((ch=fgetc(fp)) != '\n' && ch != EOF) {}
            ds->skipped++; continue;
        }
        float values[NF]; int label;
        if (dataset_parse_training_row(line, values, &label) != 0) { ds->skipped++; continue; }
        if (ds_push(ds, values, label)) { fclose(fp); dataset_free(ds); return -1; }
    }
    fclose(fp);
    return ds->n;
}

void dataset_free(Dataset *ds) {
    free(ds->raw);
    free(ds->label);
    memset(ds, 0, sizeof *ds);
}

int dataset_split(const Dataset *ds, Dataset *train, Dataset *test, int test_per_10) {
    memset(train, 0, sizeof *train);
    memset(test, 0, sizeof *test);
    for (int i = 0; i < ds->n; i++) {
        Dataset *dst = ((i % 10) < test_per_10) ? test : train;
        if (ds_push(dst, &ds->raw[i * NF], ds->label[i]) != 0) { dataset_free(train); dataset_free(test); return -1; }
    }
    return 0;
}

/* Provenance rows must be in exactly the same order as training rows. */
int dataset_split_provenance(const Dataset *ds, const char *path, Dataset *train,
                             Dataset *test, int by_station) {
    memset(train, 0, sizeof *train); memset(test, 0, sizeof *test);
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[1024];
    if (!fgets(line, sizeof line, fp) ||
        strncmp(line, "station_id,date,published_aqi,published_category", strlen("station_id,date,published_aqi,published_category")) != 0) {
        fclose(fp); return -1;
    }
    int i = 0;
    while (fgets(line, sizeof line, fp)) {
        if (i >= ds->n) { i = -1; break; }
        char station[64], date[32], category[80]; float aqi;
        if (sscanf(line, "%63[^,],%31[^,],%f,%79[^\r\n]", station, date, &aqi, category) != 4 ||
            (!isfinite(aqi) || aqi < 0) ||
            strlen(date) != 10 || date[4] != '-' || date[7] != '-' ||
            ((strcmp(category, "Moderate") == 0 ? 2 : label_from_string(category)) != ds->label[i])) break;
        unsigned long hash = 5381;
        for (const unsigned char *c = (const unsigned char *)station; *c; c++)
            hash = hash * 33u + *c;
        int is_test = by_station ? (hash % 5u == 0u) : (strcmp(date, "2020-01-01") >= 0);
        Dataset *dest = is_test ? test : train;
        if (ds_push(dest, ds->raw + (size_t)i * NF, ds->label[i])) break;
        i++;
    }
    int extra = (i < 0) || (fgets(line, sizeof line, fp) != NULL);
    fclose(fp);
    if (i != ds->n || extra || train->n == 0 || test->n == 0) {
        dataset_free(train); dataset_free(test); return -1;
    }
    return 0;
}
