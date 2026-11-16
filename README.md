# Air Quality Indicator - C + kNN (historical measurements)

## Overview
This first-semester C project calculates an Indian AQI-style value and uses a distance-weighted k-nearest-neighbours classifier to predict one of six published AQI categories. It uses **seven user-entered pollutants**: PM2.5, PM10, NO2, SO2, CO, O3 and NH3. **Pb (lead) is not requested or used anywhere in the active model**, because the available historical dataset has no Pb values. The Indian AQI standard itself still includes Pb; this educational implementation does not cover all eight official pollutants.

## Historical data
The default `data/real_training.csv` now contains **all 48,086 station-day observations** in the supplied 2015-2020 archive with nonnegative, finite measurements for all seven pollutants and a recognized published AQI category. No Pb value is required or invented. The source `Moderate` label is called `Moderately polluted` in the program. `data/real_training_provenance.csv` preserves source station IDs, dates, published AQI values and original categories in matching row order. The records retain their original class distribution (no artificial balancing). Source: uploaded historical third-party CPCB-derived `station_day.csv`; individual measurements have not been independently authenticated against CPCB. The source ZIP has 108,035 station-day records; 21,010 lack a recognized label and 38,939 of the remaining records have incomplete/invalid seven-pollutant measurements.

The dataset contains daily summaries, which do not establish the exact CPCB averaging periods (CO and O3 require 8-hour averages), data capture completeness or official reporting eligibility. The kNN classifier uses the dataset's **published categories**, while the calculator uses CPCB concentration breakpoints. Their results can differ. This is an educational demonstration, not an official AQI reporting service or future AQI forecast.

## Improvements in this version
- Removed Pb from user input, training CSV, demo readings and sensor log.
- Expanded to all 48,086 complete seven-pollutant, labelled station-day rows (no class balancing).
- Added distance-weighted kNN, which gives closer neighbours more influence, with exact-match handling.
- Tests candidate k values 1, 3, 5, 7, 9, 11, 13 and 15. To avoid extremely slow quadratic leave-one-out calculations, k tuning and the interactive LOOCV/benchmark reports use 1,200 evenly spaced readings. Actual predictions use all 48,086 records. In hold-out evaluation, k tuning uses 1,200 samples drawn from the training portion only.
- Fits min-max normalization on the training portion for the hold-out test, avoiding test-data leakage.
- Shows majority-vote baseline and weighted-kNN hold-out accuracy on the same test records.
- Reports per-category precision/recall, macro-F1 and overall accuracy.

**Previously reproduced evaluations (weighted kNN; stronger generalization tests are about 76%):** row-interleaved 33,659 train / 14,427 test: **79.48% accuracy, 78.21% macro-F1**; unseen-station 37,973 train / 10,113 test: **76.11% accuracy, 74.87% macro-F1**; chronological 2015-2019 training (36,821) / 2020 test (11,265): **76.74% accuracy, 73.75% macro-F1**. k is selected using training-only samples. None is full-dataset LOOCV.

**Evaluation limitation:** the row-interleaved hold-out contains overlapping stations and dates. It is useful for classroom comparison, but not an unbiased test of performance on future dates or unseen stations. The test set is not used to choose k; however, repeatedly changing model settings after viewing test results would still overfit it.

## Compile and run
From the `knn_engine` folder with GCC and Make:

```sh
make
make test
make demo
./knn_engine
```

Input order: `PM2.5 PM10 NO2 SO2 CO O3 NH3`. CO is mg/m3; the others are ug/m3. For a meaningful AQI calculation, use 24-hour averages except CO and O3 (8-hour averages). This console model requires all seven inputs for kNN. Missing input values cannot be classified by kNN; the AQI calculator can independently handle partial pollutant coverage in the source code.

Training CSV header: `pm25,pm10,no2,so2,co,o3,nh3,label`. The supported labels are Good, Satisfactory, Moderately polluted, Poor, Very Poor and Severe. Sensor log header: `pm25,pm10,no2,so2,co,o3,nh3`.

## Project files
- `src/aqi.c`: AQI-style subindex calculation, seven supported pollutants
- `src/knn.c`: fixed-point, floating-point and distance-weighted kNN
- `src/normalize.c`: min-max scaling
- `src/dataset.c`: position-preserving CSV parsing, validation and three evaluation splits
- `src/metrics.c`: confusion matrix, precision, recall, macro-F1
- `src/main.c`: menu, comparisons, manual readings and batch readings
- `data/real_training.csv`: 48,086 eligible historical station-day records
- `data/real_training_provenance.csv`: matching original station/date metadata
- `data/illustrative_training.csv`: artificial example readings (not official)
- `data/sensor_log.csv`: artificial demo inputs
- `docs/DEVELOPMENT_LOG.md`: development and testing notes
- `legacy/`: bundled historical original files (may contain Pb as part of preserved history)

## References
- CPCB AQI methodology: https://airquality.cpcb.gov.in/ccr_docs/About_AQI.pdf
- Government of India data portal: https://www.data.gov.in/resource/real-time-air-quality-index-various-locations

## Reproducibility and evaluation fixes (November 2026)
- The **complete project ZIP** includes all source modules, header files, Makefile, tests and provenance CSV. Do not compile the earlier six-file excerpt as a standalone program.
- CSV rows with blank fields, extra fields, NaN, infinities or negative concentrations are rejected rather than silently shifting columns. Batch sensor input follows the same nonnegative, finite, seven-value rule as kNN input.
- Normalization experiment compares raw and normalized **float kNN on the exact same 1,200 observations**, same labels and same k. Normalized 75.25% vs raw 75.08% on this sample. This small difference is not proof that normalization always improves accuracy.
- Menu options 8 and 9 evaluate unseen stations and the future year 2020, using aligned station/date provenance; option 7 retains the original interleaved baseline.
- Provenance must match the training CSV row for row. The supplied file does; if you replace the training CSV, regenerate provenance before station/date testing. Splitting uses deterministic station-ID hashing (one fifth of station IDs, not necessarily one fifth of rows) or date >= 2020-01-01.
- **34 automated assertions passed**, plus compilation and full demo. Results saved in `docs/VERIFIED_TEST_RESULTS.txt`.
- The complete station/date tests do not guarantee generalization to every location or season. Published daily category labels and direct CPCB-style calculations can differ.

## Final audit notes
- `docs/DATASET_DUPLICATE_AUDIT.md` explains why 74 identical feature/label rows were retained: station/date pairs are unique.
- `docs/VERIFIED_TEST_RESULTS.txt` contains the reproduced earlier benchmark results. The latest input-validation regression tests are separate from those benchmarks.
- The seven CPCB breakpoint endpoints match the published national AQI table (https://airquality.cpcb.gov.in/ccr_docs/About_AQI.pdf). The calculator uses simplified interpolation and caps above the highest specified breakpoint at 500; it is not an official AQI certification tool.
- The complete ZIP includes the Word development log and the legacy folder. Standalone copies are supplied for convenience.


## Final console-use notes (15 November 2026)

- Run `make`, then `./knn_engine` from this directory. `make test` runs the regression checks; `make demo` runs the full evaluation. Windows users need a compatible GCC/Make environment.
- Menu options must be whole numbers 0-9. An invalid or overlong entry is rejected without quitting.
- Enter seven space-separated pollutant concentrations: PM2.5, PM10, NO2, SO2, CO, O3, NH3. CO is mg/m3; all others are ug/m3. Use 24-hour averages except CO and O3 (8-hour averages).
- `-1` denotes missing typed input: the program then shows only the rule-based AQI if the minimum-data condition is met. kNN needs all seven nonnegative readings. Batch CSV requires seven valid nonnegative fields.
- Values above 1,000,000 or nonfinite values are rejected. Values outside the observed training range are clamped for kNN, with a warning.
- The calculator caps values beyond the highest listed pollutant breakpoint at AQI 500 (Severe); this is an educational handling rule, **not an official CPCB extrapolation**. Exact averaging-period validity and measurement completeness cannot be verified from user-entered numbers alone.
- The kNN prediction is a statistical estimate and may disagree with the calculated AQI. Neither output is a certified real-time safety assessment.
- The provenance file supplies station/date/category for evaluation, but does not repeat the seven pollutant measurements. Therefore the loader checks row count and labels; it cannot independently prove full row-to-source matching.
- The `legacy/` folder is historical material, not needed to build the active application.
