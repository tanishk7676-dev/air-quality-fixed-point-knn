#include <stdio.h>
#include <assert.h>
#include "aqi.h"
#include "dataset.h"
#include "normalize.h"
#include "knn.h"
static int tests=0;
#define CHECK(x) do { if(!(x)) {fprintf(stderr,"FAIL: %s at line %d\n",#x,__LINE__);return 1;} tests++; } while(0)
int main(void) {
 float x[NF]={30,50,40,40,1,50,200}; int dominant=-1;
 CHECK(aqi_calculate(x,&dominant)==50);
 CHECK(aqi_category(50)==0); CHECK(aqi_category(51)==1);
 CHECK(aqi_category(100)==1); CHECK(aqi_category(101)==2);
 CHECK(aqi_category(200)==2); CHECK(aqi_category(201)==3);
 CHECK(aqi_category(300)==3); CHECK(aqi_category(301)==4);
 CHECK(aqi_category(400)==4); CHECK(aqi_category(401)==5);
 x[0]=60; CHECK(aqi_calculate(x,&dominant)==100); CHECK(dominant==0);
 x[0]=90; CHECK(aqi_calculate(x,&dominant)==200);
 x[0]=120; CHECK(aqi_calculate(x,&dominant)==300);
 x[0]=250; CHECK(aqi_calculate(x,&dominant)==400);
 x[0]=300; CHECK(aqi_calculate(x,&dominant)==500);
 for(int j=0;j<NF;j++) x[j]=-1;
 CHECK(aqi_calculate(x,&dominant)==-1);
 x[2]=10;x[3]=10;x[4]=0.1;CHECK(aqi_calculate(x,&dominant)==-1);
 x[0]=20;CHECK(aqi_calculate(x,&dominant)>=0);
 Dataset ds={0}; CHECK(dataset_load(&ds,"data/illustrative_training.csv")>200);
 CHECK(ds.skipped==0);
 Scaler sc;scaler_fit(&sc,&ds);Prepared p={0};CHECK(prepared_build(&p,&ds,&sc)==0);
 int16_t q[NF];for(int j=0;j<NF;j++)q[j]=scaler_q(&sc,ds.raw[j],j);
 int pred=knn_predict_fixed(p.q,p.label,p.n,q,3,-1,NULL);
 CHECK(pred>=0 && pred<NCLASS);
 float parsed[NF]; int lbl;
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,7",parsed)==0);
 CHECK(dataset_parse_sensor_row("1,,3,4,5,6,7",parsed)!=0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,-2",parsed)!=0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,nan",parsed)!=0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,7,8",parsed)!=0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,1e999",parsed)!=0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,-1",parsed)!=0);
 CHECK(dataset_parse_training_row("1,2,3,4,5,6,7,",parsed,&lbl)!=0);
 CHECK(dataset_parse_training_row("1,2,3,4,5,6,7,Good",parsed,&lbl)==0 && lbl==0);
 CHECK(dataset_parse_training_row("1,,3,4,5,6,7,Good",parsed,&lbl)!=0);
 /* Regression: large but finite scaler values must not overflow fixed-point math. */
 Dataset extreme={0}; float samples[2*NF];
 for (int j=0;j<NF;j++) { samples[j]=0.0f; samples[NF+j]=1000000.0f; }
 extreme.raw=samples; extreme.n=2;
 Scaler big; scaler_fit(&big,&extreme);
 CHECK(scaler_q(&big,0.0f,0)==0);
 CHECK(scaler_q(&big,1000000.0f,0)==SCALE);
 CHECK(scaler_q(&big,500000.0f,0)>=SCALE/2-1);
 CHECK(scaler_q(&big,1000001.0f,0)==SCALE);
 CHECK(scaler_q(&big,-100.0f,0)==0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,7 ",parsed)==0);
 CHECK(dataset_parse_sensor_row("1,2,3,4,5,6,7,",parsed)!=0);
 prepared_free(&p);dataset_free(&ds);
 printf("PASS: %d checks\n",tests);return 0;
}
