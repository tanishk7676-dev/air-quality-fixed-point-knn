#include "aqi.h"
#include <math.h>
/* CPCB AQI upper concentration breakpoints; order: PM2.5 PM10 NO2 SO2 CO O3 NH3. */
static const double limits[NF][5] = {
    {30,60,90,120,250}, {50,100,250,350,430},
    {40,80,180,280,400}, {40,80,380,800,1600},
    {1,2,10,17,34}, {50,100,168,208,748},
    {200,400,800,1200,1800}
};
static const double index_limits[6] = {0,50,100,200,300,400};
static double subindex(double value, int pollutant) {
    if (value <= 0) return 0;
    for (int band=0; band<5; band++) {
        if (value <= limits[pollutant][band]) {
            double low = band ? limits[pollutant][band-1] : 0;
            double high = limits[pollutant][band];
            double ilo = index_limits[band], ihi = index_limits[band+1];
            return ilo + (value-low)*(ihi-ilo)/(high-low);
        }
    }
    /* Educational upper-end policy: values above the last concentration
       breakpoint are marked Severe (500). This is a deliberate cap,
       not a CPCB-approved extrapolation formula. */
    return 500;
}
int aqi_category(int value) {
    if (value < 0) return -1;
    if (value <= 50) return 0;
    if (value <= 100) return 1;
    if (value <= 200) return 2;
    if (value <= 300) return 3;
    if (value <= 400) return 4;
    return 5;
}
int aqi_calculate(const float readings[NF], int *dominant) {
    int count=0, has_pm=0, worst=-1;
    double max=-1;
    for (int j=0;j<NF;j++) {
        double v=readings[j];
        if (!isfinite(v) || v < 0) continue;
        count++;
        if (j<=1) has_pm=1;
        double sub=subindex(v,j);
        if (sub>max) {max=sub; worst=j;}
    }
    if (dominant) *dominant=worst;
    if (count<3 || !has_pm) return -1;
    int result=(int)lround(max);
    return result>500 ? 500 : result;
}
