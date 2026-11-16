#ifndef AQI_H
#define AQI_H
#include "config.h"
/* Inputs must already be valid 24-hour averages, except CO and O3 (8-hour). */
/* Missing inputs use -1. Return -1 when CPCB minimum pollutant rule fails. */
int aqi_calculate(const float readings[NF], int *dominant);
int aqi_category(int aqi);
#endif
