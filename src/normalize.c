/* normalize.c */
#include "normalize.h"

void scaler_fit(Scaler *sc, const Dataset *ds) {
    for (int j = 0; j < NF; j++) {
        float lo = ds->raw[j], hi = ds->raw[j];
        for (int i = 1; i < ds->n; i++) {
            float v = ds->raw[i * NF + j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        sc->fmin[j] = lo;            sc->fmax[j] = hi;
        sc->qmin[j] = 0;
        sc->qmax[j] = SCALE;
    }
}

int16_t scaler_q(const Scaler *sc, float v, int j) {
    float range = sc->fmax[j] - sc->fmin[j];
    if (range <= 0.0f) return 0;
    if (v <= sc->fmin[j]) return 0;
    if (v >= sc->fmax[j]) return SCALE;
    float normalized = (v - sc->fmin[j]) / range;
    return (int16_t)(normalized * SCALE);
}

float scaler_f(const Scaler *sc, float v, int j) {
    if (v < sc->fmin[j]) v = sc->fmin[j];
    if (v > sc->fmax[j]) v = sc->fmax[j];
    float range = sc->fmax[j] - sc->fmin[j];
    if (range <= 0.0f) return 0.0f;
    return (v - sc->fmin[j]) / range;
}

int scaler_in_range(const Scaler *sc, float v, int j) {
    return v >= sc->fmin[j] && v <= sc->fmax[j];
}
