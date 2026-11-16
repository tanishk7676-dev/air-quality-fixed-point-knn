/* gen_data.c - generates a synthetic, labelled air-quality dataset.
 * Usage: gen_data <rows> > data/air_quality.csv
 * Columns: co2 (ppm), co (ppm), temp (C), humidity (%), label
 * Label comes from a weighted "pollution index" plus sensor noise,
 * so classes overlap a little (like real data).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static uint32_t state = 20260311u;                 /* fixed seed -> reproducible */
static double rnd(void) {                          /* simple LCG, returns 0..1 */
    state = state * 1664525u + 1013904223u;
    return (state >> 8) / 16777216.0;
}

int main(int argc, char **argv) {
    int rows = (argc > 1) ? atoi(argv[1]) : 600;
    const char *names[3] = {"Clean", "Moderate", "Hazardous"};
    printf("co2,co,temp,humidity,label\n");
    for (int i = 0; i < rows; i++) {
        double co2  = 400.0 + 2100.0 * pow(rnd(), 1.3);
        double co   = 25.0 * pow(rnd(), 1.5);
        double temp = 15.0 + 25.0 * rnd();
        double hum  = 20.0 + 70.0 * rnd();
        double idx  = 0.4 * (co2 - 400.0) / 2100.0 + 0.4 * co / 25.0
                    + 0.1 * (hum - 20.0) / 70.0   + 0.1 * (temp - 15.0) / 25.0;
        idx += (rnd() - 0.5) * 0.08;               /* sensor noise */
        int label = (idx < 0.32) ? 0 : (idx < 0.52) ? 1 : 2;
        printf("%.0f,%.1f,%.1f,%.1f,%s\n", co2, co, temp, hum, names[label]);
    }
    return 0;
}
