/* knn.h - k-nearest-neighbour engine (fixed-point and float versions) */
#ifndef KNN_H
#define KNN_H

#include "config.h"
#include "dataset.h"
#include "normalize.h"

/* A normalised training set, held in BOTH formats so they can be compared. */
typedef struct {
    int      n;
    int16_t *q;        /* n * NF fixed-point values, 0..256 */
    float   *f;        /* n * NF float values, 0..1         */
    int     *label;    /* n labels (copied)                 */
} Prepared;

int  prepared_build(Prepared *p, const Dataset *ds, const Scaler *sc);
void prepared_free(Prepared *p);

/* Predict one query. skip = index to ignore (leave-one-out) or -1.
 * votes_out (optional) receives how many of the k neighbours agreed.   */
int knn_predict_fixed(const int16_t *train, const int *label, int n,
                      const int16_t *q, int k, int skip, int *votes_out);
int knn_predict_float(const float *train, const int *label, int n,
                      const float *q, int k, int skip, int *votes_out);

/* Leave-one-out accuracy in %. pred (optional, length n) receives predictions. */
float knn_loo_fixed(const int16_t *train, const int *label, int n, int k, int *pred);
float knn_loo_float(const float *train, const int *label, int n, int k, int *pred);

/* Adaptive step: try every candidate k, return the best one.
 * Ties go to the smaller k (cheaper on a microcontroller).
 * acc_out (optional, length nc) receives the accuracy of each candidate. */
int knn_select_k(const int16_t *train, const int *label, int n,
                 const int *cands, int nc, float *acc_out);

int knn_predict_weighted_fixed(const int16_t *train, const int *label, int n,
                      const int16_t *q, int k, int skip, int *votes_out);
float knn_loo_weighted_fixed(const int16_t *train, const int *label, int n, int k, int *pred);
int knn_select_weighted_k(const int16_t *train, const int *label, int n,
                 const int *cands, int nc, float *acc_out);
#endif
