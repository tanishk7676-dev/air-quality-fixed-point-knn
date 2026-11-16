# Development Log — Adaptive Fixed-Point kNN Engine

**Project:** An Adaptive Fixed-Point k-Nearest Neighbour Engine for Low-Power Sensor Classification
**Use case:** Indoor air-quality indicator (Clean / Moderate / Hazardous)
**Built with:** Claude (AI assistant) in a staged build–test–fix workflow

## How to read this log

Every entry below records something that was **actually compiled and run**. The raw
compiler/program output for each stage is saved in the `history/` folder, so each claim
can be checked. Nothing here is a reconstructed or imagined mistake: the defects were
found by running the code, the compiler warnings, the sanitizers (`-fsanitize=address,undefined`)
and by reviewing the program's own output.

The workflow was: **write a draft → compile → run → test edge cases → diagnose → fix → re-test.**

---

## Version summary

| Version | File(s) | What it was | Result |
|---|---|---|---|
| v1 | `history/knn_v1.c` | First complete single-file draft of the whole pipeline | Ran, but results were wrong (k=1 gave 100%) |
| v2 | `history/knn_v2.c` | Fixed leave-one-out bug and compiler warning | Realistic accuracy, but edge cases crashed |
| v3 | `history/knn_v3.c` | Fixed overflow, divide-by-zero, k overflow, duplicated code | All edge-case checks pass |
| v4 | `src/*` (modular) | Rewrote as modules + menu, benchmark, hold-out, confusion matrix | Compiled clean; review found 2 more problems |
| v5 | `src/*` (final) | Fixed benchmark wording, widened k search, added unit tests | 33/33 tests pass, sanitizer-clean |

---

## v1 — First draft

**Goal:** the whole pipeline in one file: load CSV → Q8 fixed-point → min-max normalise → kNN →
leave-one-out (LOO) validation → adaptive choice of k → classify a new reading, with a float
version alongside for comparison.

**Ran:** `gcc -Wall -Wextra -O2 history/knn_v1.c` then the program.

**What happened** (`history/v1_run_output.txt`, `history/v1_compile_output.txt`):

```
 k | fixed % | float %
 1 | 100.00  | 100.00
 3 |  93.83  |  93.67
 5 |  93.17  |  93.00
 7 |  92.17  |  92.00
Adaptive choice: k = 1 (accuracy 100.00%)
```

Compiler warning: `ignoring return value of 'fgets'`.

**Diagnosis**

- **Defect D1 — leave-one-out included the sample itself.** 100% accuracy at k=1 on noisy data
  is not believable. The classifier searched the *whole* training set, which still contained the
  query point at distance 0, so every point "found" itself. This also inflated k=3, 5, 7, and made
  the adaptive selector always choose k=1.
- **Defect D2 — unchecked `fgets` return value** (compiler warning).

---

## v2 — Fix D1 and D2

**Change:** added a `skip` index to the classify functions so LOO ignores the sample being tested;
checked the return value of `fgets`.

**Result** (`history/v2_run_output.txt`): accuracy dropped to believable values.

```
 1 | 85.83 | 86.00      3 | 88.67 | 88.83
 5 | 88.50 | 88.67      7 | 88.33 | 88.33
Adaptive choice: k = 3 (accuracy 88.67%)
```

The compiler warning disappeared. The program looked correct, so it was **stress-tested** with
`history/probe_v2.c` built with AddressSanitizer + UBSan (`history/v2_probe_output.txt`):

| Test | Result on v2 | Expected |
|---|---|---|
| CO2 = 65535 (sensor glitch) | `signed integer overflow: 16674560 * 256`, value became **-49** | clamp to 256 |
| CO2 = 100 (below training range) | became **-36** | clamp to 0 |
| k = 20 | **AddressSanitizer: stack-buffer-overflow** in `classify_fixed` | safe handling |
| Dead sensor (constant column) | **crash: division by zero** in `normalise_all` (`history/v2_constant_column_output.txt`) | no crash |
| Tied votes (k=3, k=5) | **0 of 600** samples had a tie | (hypothesis check) |

**Diagnosis**

- **D3 — integer overflow / no clamping** when a reading is outside the training range.
- **D4 — stack buffer overflow** when k > the 16-slot internal array.
- **D5 — divide-by-zero** when a feature has max == min (e.g. a failed sensor outputting a constant).
- **D6 — duplicated code:** the normalisation formula was copy-pasted into `main()`, so any fix
  would have to be made in two places.
- **Suspected but not real:** I suspected vote ties (three classes, odd k) would be a problem. The
  probe showed 0 ties on this dataset, so this is *not* a defect for this data. A nearest-neighbour
  tie-break was still added as a cheap defensive measure, because other datasets could tie.

---

## v3 — Fix D3–D6

**Changes:** one shared `normalise_q()` / `normalise_f()` that clamps to the training range and
returns 0 for zero-range features; `MAX_K` bound on k; unfilled neighbour slots ignored;
nearest-neighbour tie-break.

**Result:** `history/probe_v3.c` re-runs every failing case as PASS/FAIL (`history/v3_probe_output.txt`):
all 6 checks pass, under the sanitizers. Normal accuracy is unchanged from v2 (88.67%), confirming
the fixes did not alter normal behaviour. The dead-sensor CSV that crashed v2 now runs
(`history/v3_constant_column_output.txt`; accuracy falls to ~50%, as expected, because the CO
information is gone).

**A further defect found by code review, then confirmed** (`history/v3_label_probe_output.txt`):

- **D7 — unknown labels silently became "Hazardous".** `label_from_string()` returned 2 for *any*
  text it didn't recognise. Probe result: `"Hazardus"` (typo) → 2, `"banana"` → 2, and even
  `"clean"` in lowercase → **2 (Hazardous)**. A small typo in a data file would silently corrupt
  the training labels.

---

## v4 — Modular rewrite

**Goal:** turn the single file into a proper project: `config.h`, `dataset`, `normalize`, `knn`,
`metrics`, `main`; dynamic memory (`malloc`/`realloc`) for the dataset; menu + `--demo` mode;
confusion matrix with precision/recall; float-vs-fixed benchmark; a hold-out test (fit scaler and
choose k on training data only, score on unseen data); a "why normalise" experiment; out-of-range
warnings. D7 was fixed here (`label_from_string` returns -1; bad lines are skipped and counted).

**Result:** compiled with `-std=c99 -Wall -Wextra -Wpedantic` with no warnings; the whole demo ran
under the sanitizers with no errors (`history/v4_demo_output.txt`). Reading the *output* critically
still showed two problems of my own making:

- **D8 — the benchmark text overclaimed.** It printed that speed on a PC is "similar", but the
  measured numbers showed fixed-point **slower** (308 ms vs 243 ms). A PC has a hardware FPU, so
  fixed-point cannot show a speed advantage here. The message was making a claim the data did not support.
- **D9 — the adaptive search stopped at its own edge.** k candidates were {1…11} and the chosen
  value was 11, the last one. Accuracy might have kept rising beyond it.

(Snapshot of this version's `main.c`: `history/v4_main.c`.)

---

## v5 — Final

**Changes:** benchmark message rewritten to state only what is measured (memory saved; fixed-point
shows no speed gain on an FPU machine; the MCU speed benefit is *not measured here*). k candidates
widened to {1,3,5,…,15} (= `MAX_K`).

**Result** (`history/v5_demo_output.txt`): with the wider search, k = 11 is still the best
(k=13 → 89.67%, k=15 → 89.17%), so 11 is a genuine peak, not an artefact of the search limit.

**Unit tests** (`tests/test_knn.c`, `history/v5_test_output.txt`): 33 checks covering every
defect above, plus the engine's core behaviour. All pass.

**Do the tests really catch the bugs?** A mutation check was run (`history/v5_mutation_check_output.txt`):
the v1 bug (removing the `if (i == skip) continue;` line) was re-introduced into a copy, and exactly
the two relevant tests failed (`LOO excludes the sample itself`, `accuracy is realistic`). The suite
is also clean under AddressSanitizer + UBSan, with no leaks.

**A note on timings.** The timing numbers in `history/v4_demo_output.txt` and `v5_demo_output.txt`
came from the *sanitizer* build, which is deliberately slow, so they are **not** representative.
The valid timings are from the `-O2` build in `history/final_demo_output.txt`.

---

## Defect register

| ID | Defect | Found by | Fixed in | Regression test |
|---|---|---|---|---|
| D1 | LOO counted the sample against itself (k=1 → 100%) | Suspicious result | v2 | `LOO excludes the sample itself (0%)` |
| D2 | Unchecked `fgets` return | Compiler warning | v2 | — (compiler flags) |
| D3 | Overflow / no clamp on out-of-range reading | UBSan probe | v3 | `65535 clamps to 256`, `1e30 clamps` |
| D4 | Stack overflow for k > 16 | AddressSanitizer | v3 | `k=100 is clamped` |
| D5 | Divide-by-zero on constant column | UBSan / SIGFPE | v3 | `constant column maps to 0` |
| D6 | Duplicated normalisation code | Code review | v3 | (single shared function) |
| D7 | Unknown / lowercase labels became "Hazardous" | Code review + probe | v4 | 5 label-parsing tests |
| D8 | Benchmark message overclaimed speed | Reading the output | v5 | — (output wording) |
| D9 | k search ended at its own boundary | Reading the output | v5 | — (candidate list) |

---

## Final results (from `history/final_demo_output.txt`, `-O2` build)

| Measure | Value |
|---|---|
| Dataset | 600 synthetic readings (163 Clean / 255 Moderate / 182 Hazardous) |
| Adaptive k chosen | 11 |
| Leave-one-out accuracy, fixed-point (Q8) | 90.17% |
| Leave-one-out accuracy, float | 90.00% |
| Predictions that differ, fixed vs float | 1 of 600 |
| **Hold-out accuracy on 180 unseen samples** | **85.00%** |
| Feature memory, float → fixed-point | 9600 B → 4800 B (50% saved) |
| Time on PC (FPU present), float vs fixed | 19.7 ms vs 23.8 ms (fixed is *not* faster here) |
| Raw features vs normalised (float) | 58.67% → 90.00% |

The **hold-out figure (85.00%) is the more honest estimate** of performance on new data; the LOO
figure is slightly optimistic because k was tuned on the same data.

## Known limitations (stated honestly)

1. **The dataset is synthetic.** Labels come from a formula plus noise, so the results show that the
   *engine works*, not how accurate it would be on real air-quality data. For a stronger report, test
   on a real dataset (e.g. UCI) — the CSV format and `config.h` constants are easy to adapt.
2. **The MCU speed advantage was not measured.** Only the 50% memory saving is demonstrated. No
   microcontroller was used.
3. **Glitch values** such as 65535 are clamped and (conservatively) classed as Hazardous *with a
   warning to check the sensor*. A production system might reject them outright as faults.
4. The hold-out run chose k = 15, the edge of the candidate list; a larger `MAX_K` might do slightly
   differently.
5. The whole training set is kept in memory and every prediction scans all of it. On a real MCU a
   smaller prototype set would be needed (future work).
6. Parsing and reporting use `float`; the classification path itself uses integers. An MCU port would
   feed integer ADC readings straight into the scaler.

## Prompt history (the real conversation)

1. Gave the project title; asked what kNN is and for projects feasible at semester-1 C level.
2. Asked what combining projects 1, 2, 3 and 7 would produce.
3. Asked what useful application these pieces could make for PBL.
4. Asked exactly how the gas / air-quality indicator would work (inputs, outputs, process).
5. Asked the AI to build the whole project, showing a trial-and-error history of AI mistakes that were
   then corrected.

## How to reproduce everything

```
make            # build engine + data generator
make test       # 33 unit tests
make demo       # full non-interactive demo
make debug      # sanitizer build (memory / undefined-behaviour checks)
./gen_data 600 > data/air_quality.csv   # regenerate the dataset (fixed seed)
```

## Student's own notes (to be completed by the team)

*Fill this in truthfully — evaluators value your own understanding.*

- What I ran myself and on what machine / compiler:
- What I changed or experimented with (e.g. different k candidates, different thresholds, real dataset):
- Something I did not understand at first and how I worked it out:
- What I would improve next:
