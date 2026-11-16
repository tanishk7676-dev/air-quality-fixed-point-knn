/* normalize.h - fixed-point min-max scaling to the range 0..256 */
#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "config.h"
#include "dataset.h"

typedef struct {
    float   fmin[NF], fmax[NF];    /* training min / max (float)        */
    int32_t qmin[NF], qmax[NF];    /* same, in Q8 fixed-point           */
} Scaler;

void    scaler_fit(Scaler *sc, const Dataset *ds);

/* both clamp to the training range; a constant (dead) sensor maps to 0 */
int16_t scaler_q(const Scaler *sc, float v, int j);   /* fixed-point 0..256 */
float   scaler_f(const Scaler *sc, float v, int j);   /* float 0..1         */

/* 1 if v lies inside the range seen during training */
int     scaler_in_range(const Scaler *sc, float v, int j);

#endif
