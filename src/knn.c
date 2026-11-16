/* knn.c */
#include "knn.h"
#include <stdlib.h>
#include <limits.h>

int prepared_build(Prepared *p, const Dataset *ds, const Scaler *sc) {
    p->n = ds->n;
    p->q = malloc((size_t)ds->n * NF * sizeof(int16_t));
    p->f = malloc((size_t)ds->n * NF * sizeof(float));
    p->label = malloc((size_t)ds->n * sizeof(int));
    if (!p->q || !p->f || !p->label) { prepared_free(p); return -1; }
    for (int i = 0; i < ds->n; i++) {
        p->label[i] = ds->label[i];
        for (int j = 0; j < NF; j++) {
            p->q[i * NF + j] = scaler_q(sc, ds->raw[i * NF + j], j);
            p->f[i * NF + j] = scaler_f(sc, ds->raw[i * NF + j], j);
        }
    }
    return 0;
}

void prepared_free(Prepared *p) {
    free(p->q); free(p->f); free(p->label);
    p->q = NULL; p->f = NULL; p->label = NULL; p->n = 0;
}

/* Majority vote over neighbour labels sorted nearest-first.
 * A tie is broken in favour of the class of the nearest tied neighbour. */
static int vote(const int *lab, int k, int *winner_votes) {
    int votes[NCLASS] = {0}, top = 0;
    for (int i = 0; i < k; i++) if (lab[i] >= 0) votes[lab[i]]++;
    for (int c = 0; c < NCLASS; c++) if (votes[c] > top) top = votes[c];
    if (winner_votes) *winner_votes = top;
    for (int i = 0; i < k; i++)
        if (lab[i] >= 0 && votes[lab[i]] == top) return lab[i];
    return 0;
}

static int clamp_k(int k) { return k < 1 ? 1 : (k > MAX_K ? MAX_K : k); }

int knn_predict_fixed(const int16_t *train, const int *label, int n,
                      const int16_t *q, int k, int skip, int *votes_out) {
    k = clamp_k(k);
    int32_t best[MAX_K]; int lab[MAX_K];
    for (int i = 0; i < k; i++) { best[i] = INT32_MAX; lab[i] = -1; }
    for (int i = 0; i < n; i++) {
        if (i == skip) continue;
        int32_t d = 0;                                   /* squared distance, no sqrt needed */
        for (int j = 0; j < NF; j++) {
            int32_t diff = (int32_t)train[i * NF + j] - q[j];
            d += diff * diff;
        }
        int p = k - 1;
        if (d < best[p]) {                               /* insert into sorted list */
            while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; lab[p] = lab[p - 1]; p--; }
            best[p] = d; lab[p] = label[i];
        }
    }
    return vote(lab, k, votes_out);
}

int knn_predict_float(const float *train, const int *label, int n,
                      const float *q, int k, int skip, int *votes_out) {
    k = clamp_k(k);
    float best[MAX_K]; int lab[MAX_K];
    for (int i = 0; i < k; i++) { best[i] = 1e30f; lab[i] = -1; }
    for (int i = 0; i < n; i++) {
        if (i == skip) continue;
        float d = 0;
        for (int j = 0; j < NF; j++) {
            float diff = train[i * NF + j] - q[j];
            d += diff * diff;
        }
        int p = k - 1;
        if (d < best[p]) {
            while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; lab[p] = lab[p - 1]; p--; }
            best[p] = d; lab[p] = label[i];
        }
    }
    return vote(lab, k, votes_out);
}

float knn_loo_fixed(const int16_t *train, const int *label, int n, int k, int *pred) {
    int ok = 0;
    for (int i = 0; i < n; i++) {
        int p = knn_predict_fixed(train, label, n, &train[i * NF], k, i, NULL);
        if (pred) pred[i] = p;
        if (p == label[i]) ok++;
    }
    return n ? 100.0f * ok / n : 0.0f;
}

float knn_loo_float(const float *train, const int *label, int n, int k, int *pred) {
    int ok = 0;
    for (int i = 0; i < n; i++) {
        int p = knn_predict_float(train, label, n, &train[i * NF], k, i, NULL);
        if (pred) pred[i] = p;
        if (p == label[i]) ok++;
    }
    return n ? 100.0f * ok / n : 0.0f;
}

int knn_select_k(const int16_t *train, const int *label, int n,
                 const int *cands, int nc, float *acc_out) {
    int best_k = cands[0];
    float best_acc = -1.0f;
    for (int c = 0; c < nc; c++) {
        float a = knn_loo_fixed(train, label, n, cands[c], NULL);
        if (acc_out) acc_out[c] = a;
        if (a > best_acc) { best_acc = a; best_k = cands[c]; }   /* strict > keeps smaller k on ties */
    }
    return best_k;
}

/* Inverse-distance weighted voting; exact matches take priority. */
int knn_predict_weighted_fixed(const int16_t *train, const int *label, int n,
                      const int16_t *q, int k, int skip, int *votes_out) {
    k = clamp_k(k);
    int32_t best[MAX_K]; int lab[MAX_K];
    for (int j=0;j<k;j++) { best[j]=INT32_MAX; lab[j]=-1; }
    for (int i=0;i<n;i++) {
        if (i==skip) continue;
        int32_t d=0;
        for(int j=0;j<NF;j++) { int32_t z=(int32_t)train[i*NF+j]-q[j]; d+=z*z; }
        int pos=k-1;
        if (d<best[pos]) {
            while(pos>0 && best[pos-1]>d) {best[pos]=best[pos-1];lab[pos]=lab[pos-1];pos--;}
            best[pos]=d;lab[pos]=label[i];
        }
    }
    double scores[NCLASS]={0}; int counts[NCLASS]={0}; int exact=0;
    for(int j=0;j<k;j++) if(lab[j]>=0 && best[j]==0) exact++;
    for(int j=0;j<k;j++) if(lab[j]>=0 && (!exact || best[j]==0)) {
        scores[lab[j]] += exact ? 1.0 : 1.0/(1.0+(double)best[j]);
        counts[lab[j]]++;
    }
    int winner=0;
    for(int j=0;j<k;j++) if(lab[j]>=0 && scores[lab[j]]>scores[winner]) winner=lab[j];
    if(votes_out) *votes_out=counts[winner];
    return winner;
}
float knn_loo_weighted_fixed(const int16_t *train, const int *label, int n, int k, int *pred) {
    int correct=0;
    for(int i=0;i<n;i++) {
        int result=knn_predict_weighted_fixed(train,label,n,&train[i*NF],k,i,NULL);
        if(pred) pred[i]=result;
        if(result==label[i]) correct++;
    }
    return n ? 100.0f*correct/n : 0;
}
int knn_select_weighted_k(const int16_t *train, const int *label, int n,
                 const int *cands, int nc, float *acc_out) {
    int best=cands[0];float top=-1;
    for(int j=0;j<nc;j++) {
        float a=knn_loo_weighted_fixed(train,label,n,cands[j],NULL);
        if(acc_out) acc_out[j]=a;
        if(a>top) {top=a;best=cands[j];}
    }
    return best;
}
