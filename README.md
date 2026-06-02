# Prime Microscope

A CUDA-based experimental prime-pattern search and validation system for exploring local prime constellations at trillion-scale integer ranges.

Prime Microscope scans large ranges of integers, detects unusual local prime-offset structures, visualises prime-pattern density, automatically zooms into promising regions, validates candidate patterns using independent counting, and compares observed results against Hardy–Littlewood-style expected behaviour.

The project combines:

* GPU programming with CUDA
* Large-range prime sieving
* Prime-offset constellation detection
* Automated broad-scan and zoom-search workflows
* Statistical candidate ranking
* Prime-cluster generation
* Actual/expected anomaly hunting
* Hardy–Littlewood-style expected-count comparison
* Reproducible CSV-based experiment logging
* Pattern-space visualisation for anomaly detection

---

## Current headline result

The strongest observed family is:

```text
quad_0_6_6a_6b
```

This notation describes a family of four-prime offset patterns with the general formula:

```text
p, p + 6, p + 6a, p + 6b
```

where:

* `p` is the starting value,
* `a` and `b` are positive integers,
* `6a` and `6b` mean the later offsets are also multiples of 6.

For example:

```text
quad_0_6_126_336
```

means the program is searching for values of `p` where all four numbers are prime:

```text
p
p + 6
p + 126
p + 336
```

In this case:

```text
6a = 126, so a = 21
6b = 336, so b = 56
```

So `quad_0_6_126_336` is one exact member of the broader `quad_0_6_6a_6b` family.

---

## Why this family matters

For primes greater than 3, primes must lie in one of two residue classes:

```text
1 mod 6
5 mod 6
```

Adding a multiple of 6 keeps a number in the same residue class. Therefore, patterns of the form:

```text
p, p + 6, p + 6a, p + 6b
```

stay within the same prime-compatible modulo-6 lane.

This makes the family mathematically natural. The interesting computational result is not merely that this family exists, but that it was **heavily overrepresented among the strongest surviving candidates** produced by the search pipeline.

---

## 2-trillion holdout enrichment result

In a holdout run beginning near:

```text
2,000,000,000,000
```

the broad candidate pool contained:

```text
Total broad candidates:                  1,509,000
quad_0_6_6a_6b broad candidates:            38,280
quad_0_6_6a_6b broad frequency:          2.536779324055666%
```

After applying the strong-candidate criteria:

```text
Total strong candidates:                 12
quad_0_6_6a_6b strong candidates:         12
quad_0_6_6a_6b strong frequency:       100.0%
```

This gives an approximate enrichment factor of:

```text
39.42006269592476×
```

Interpretation:

> In this 2-trillion holdout run, a family representing only about 2.54% of broad candidates accounted for 100% of strong surviving candidates under the current scoring criteria.

This is the strongest family-level result produced by the project so far.

---

## Strong candidates from the 2-trillion run

The 2-trillion holdout run produced 12 strong offset candidates:

```text
quad_0_6_156_300
quad_0_6_132_210
quad_0_6_126_360
quad_0_6_90_330
quad_0_6_120_336
quad_0_6_66_90
quad_0_6_180_186
quad_0_6_132_252
quad_0_6_126_270
quad_0_6_66_360
quad_0_6_90_120
quad_0_6_66_150
```

All 12 match the formula:

```text
quad_0_6_6a_6b
```

This suggests that the strong-candidate filter is not returning arbitrary offset patterns. It is repeatedly selecting a specific modulo-6 structured family.

---

## Strong candidates from the 1-trillion run

A previous run beginning near:

```text
1,000,000,000,000
```

also produced strong offset candidates from the same family:

```text
quad_0_6_126_246
quad_0_6_96_210
quad_0_6_84_90
quad_0_6_132_336
quad_0_6_126_336
```

This provides an independent first-range comparison showing that the same family appeared before the 2-trillion holdout run.

There was also one notable gap motif in the 1-trillion run:

```text
gap_12_18_12
```

A gap motif means consecutive prime gaps, not merely selected prime offsets. However, this motif has not yet been confirmed as a repeated holdout finding, so it is treated as secondary.

The replicated result is the offset-family dominance:

```text
quad_0_6_6a_6b
```

---

## Best exact generator found so far

After the family-level signal was identified, several exact patterns were tested as direct prime-cluster generators.

The strongest exact generator found so far is:

```text
quad_0_6_126_336
```

This represents:

```text
p, p + 6, p + 126, p + 336
```

It ranked highest across multiple large starting ranges and remained strongest when tested using larger 10-million-number validation windows.

---

## Multi-range validation

The top exact patterns were tested across four large starting ranges:

```text
1T:   1,000,000,000,000
2T:   2,000,000,000,000
5T:   5,000,000,000,000
10T: 10,000,000,000,000
```

The purpose of this test was to check whether the strongest exact patterns remained productive as the search moved to larger number ranges.

Because primes become rarer as numbers grow, raw hit counts are expected to decrease. To compare results more fairly, a rough normalized score was calculated:

```text
normalized_score = hits_per_million × log(start)^4
```

This approximately adjusts for the expected density decay of four-prime constellations.

---

## Expanded 2-million validation ranking

The following table ranks candidate patterns using 2-million-number validation windows across 1T, 2T, 5T, and 10T.

```text
rank pattern                  avg_norm       avg_hits/M   min_hits/M   max_hits/M
1    quad_0_6_126_336       46,828,683.88       68.88        60.00        80.50
2    quad_0_6_96_210        41,454,814.41       60.62        55.50        66.50
3    quad_0_6_66_360        40,863,074.58       60.38        47.50        76.00
4    quad_0_6_120_336       39,882,415.11       57.50        48.50        65.00
5    quad_0_6_66_150        39,618,853.65       58.62        48.00        75.00
6    quad_0_6_156_300       39,196,312.31       57.25        53.50        64.00
7    quad_0_6_90_330        37,323,617.02       55.25        42.50        73.00
8    quad_0_6_126_270       37,310,544.98       54.38        49.50        58.00
9    quad_0_6_132_336       37,183,677.63       54.62        47.00        64.00
10   quad_0_6_66_90         36,906,520.99       54.25        47.00        67.00
11   quad_0_6_132_210       35,553,895.92       52.50        43.00        63.50
12   quad_0_6_90_120        35,187,728.52       51.62        44.50        66.00
13   quad_0_6_126_360       34,942,412.37       51.38        42.00        58.50
14   quad_0_6_126_246       32,510,769.93       48.00        37.00        56.00
15   quad_0_6_132_252       32,120,352.61       47.62        37.00        61.50
16   quad_0_6_84_90         29,641,111.31       43.88        36.50        58.50
```

Result:

> `quad_0_6_126_336` was the strongest exact generator among the 16 tested survivor patterns.

---

## 10-million validation of the top four patterns

The top four candidates were retested using larger 10-million-number windows.

```text
rank pattern                  avg_norm       avg_hits/M   min_hits/M   max_hits/M
1    quad_0_6_126_336       47,352,657.78       69.42        61.00        76.70
2    quad_0_6_120_336       40,709,216.11       59.75        52.40        68.60
3    quad_0_6_96_210        38,343,545.67       56.35        48.20        63.80
4    quad_0_6_66_360        37,917,091.89       55.62        46.90        61.20
```

Result:

> `quad_0_6_126_336` remained the strongest exact generator after increasing validation range size from 2 million to 10 million numbers.

This made it the leading exact pattern discovered so far.

---

## Hardy–Littlewood-style theory comparison

To check whether the strongest exact generators were unexpectedly strong, the project implemented a singular-series approximation inspired by the Hardy–Littlewood prime k-tuple heuristic.

The expected count is estimated using:

```text
expected_hits ≈ singular_series × range_size / log(x)^4
```

The comparison showed that the top patterns behave close to theoretical expectation.

For the top-four 10-million validation:

```text
rank pattern              avg_A/E   avg_actual/M   avg_exp/M   singular   ranges
1    quad_0_6_120_336       1.006        59.75        59.54      40.4843       4
2    quad_0_6_96_210        1.005        56.35        56.11      38.1487       4
3    quad_0_6_126_336       0.998        69.42        69.77      47.4426       4
4    quad_0_6_66_360        0.981        55.62        56.83      38.6441       4
```

Interpretation:

* `quad_0_6_126_336` is the best raw generator.
* Its performance is close to theory, with an average actual/expected ratio near 1.
* This suggests the pattern is strong because known prime-tuple heuristics predict it should be strong.
* It is not currently an unexplained anomaly.

---

## Expanded actual/theory comparison

Across the expanded candidate set, the highest actual/theory ratios were:

```text
rank pattern              avg_A/E   avg_actual/M   avg_exp/M   singular
1    quad_0_6_96_210        1.087        60.62        56.11      38.1487
2    quad_0_6_156_300       1.062        57.25        54.27      36.8998
3    quad_0_6_66_360        1.057        60.38        56.83      38.6441
4    quad_0_6_66_150        1.044        58.62        55.82      37.9540
5    quad_0_6_90_120        0.993        51.62        52.10      35.4238
6    quad_0_6_132_252       0.989        47.62        47.77      32.4832
7    quad_0_6_126_336       0.987        68.88        69.77      47.4426
```

Interpretation:

* `quad_0_6_126_336` remains the best raw generator.
* `quad_0_6_96_210` is the strongest mild above-theory candidate so far.
* No tested pattern currently shows a large unexplained actual/theory excess.
* Most results are consistent with Hardy–Littlewood-style expectations.

---

## Current conclusions

### 1. Family-level result

The `quad_0_6_6a_6b` family is strongly overrepresented among strong candidates.

In the 2-trillion holdout run:

```text
Broad frequency:   2.5368%
Strong frequency:  100.0%
Enrichment:        39.42×
```

This is the strongest current result.

---

### 2. Exact-generator result

The strongest exact generator tested so far is:

```text
quad_0_6_126_336
```

It produced the highest average normalized yield across 1T, 2T, 5T, and 10T, and remained strongest when tested over larger 10-million-number windows.

---

### 3. Theory comparison result

The strongest exact generator appears to be strong because the Hardy–Littlewood-style singular-series heuristic predicts it should be strong.

This indicates that the GPU search is detecting real prime-tuple structure rather than random noise, but the leading exact generator is not currently an unexplained anomaly.

---

### 4. Next-stage anomaly target

The next stage of the project is to search for exact patterns with sustained:

```text
actual / expected >= 1.2
```

The goal is to find patterns that do not merely have high raw counts, but produce more prime constellations than the Hardy–Littlewood-style estimate predicts.

A pattern with `A/E` near 1.0 is behaving close to theoretical expectation.

A pattern with sustained `A/E >= 1.2` across independent ranges would be a stronger anomaly candidate.

---

## Project architecture

The project currently consists of several main components.

### `prime2.cu`

CUDA scanner.

Responsibilities:

* Mark primes in large integer ranges.
* Count prime-offset pattern occurrences.
* Divide ranges into image tiles.
* Compute local unusualness and z-score style measures.
* Save CSV outputs.
* Save bitmap visualisations.
* Support broad scanning and candidate detection.

---

### `auto_explore.py`

Automated search driver.

Responsibilities:

* Run broad chunks.
* Select interesting patterns.
* Zoom into candidate regions.
* Track exact and related pattern survival.
* Save strong candidates.
* Continue searching after finding candidates.

---

### `pattern_prime_generator_v2.py`

Direct prime-cluster generator and validator.

Given a pattern such as:

```text
quad_0_6_126_336
```

it counts actual occurrences of:

```text
p, p+6, p+126, p+336
```

within a chosen range.

It outputs:

```text
pattern,total_hits,candidates_checked,hit_rate,hits_per_million_numbers
```

---

### `theory_compare_patterns.py`

Compares empirical pattern counts against Hardy–Littlewood-style expected values.

Outputs:

```text
actual_hits
expected_hits
actual/theory ratio
singular-series approximation
```

---

### `ae_hunter.py`

Automated actual/expected anomaly hunter.

This script tests selected patterns or generated `quad_0_6_6a_6b` family members across one or more ranges, computes expected counts using the singular-series approximation, and records the actual/expected ratio.

Its main target is:

```text
A/E >= 1.2
```

where:

```text
A/E = actual_hits / expected_hits
```

Responsibilities:

* Test candidate patterns over fixed ranges.
* Run the exact prime-cluster generator.
* Compute singular-series expected counts.
* Calculate actual/expected ratios.
* Save all results to a CSV file.
* Save special hits where `A/E` exceeds the chosen threshold.
* Continue searching after finding hits instead of stopping.

Example output files:

```text
ae_hunter_results.csv
ae_hunter_hits.csv
```

`ae_hunter_results.csv` stores all tested results.

`ae_hunter_hits.csv` stores only patterns and ranges where the A/E threshold is reached.

---

### `ae_visuals.py`

Pattern-space visualisation script.

The original bitmap outputs from the CUDA scanner can be difficult to interpret because prime-density images often appear noisy or sparse. `ae_visuals.py` creates more useful research-oriented visualisations based on actual/expected ratios and pattern-space structure.

Generated outputs include:

```text
ae_top_patterns.csv
ae_scatter_ae_vs_singular.png
ae_timeline_top_patterns.png
ae_pattern_space_ae.png
ae_pattern_space_hits_per_million.png
```

These graphics are designed to answer more useful questions:

* Which patterns are above theory?
* Which patterns are simply high-yield because theory predicts them to be?
* Which patterns stay strong across multiple ranges?
* Which areas of `quad_0_6_6a_6b` pattern space are most productive?
* Which patterns are worth deeper validation?

---

## Visualisation strategy

The first version of the project produced bitmap images from prime-pattern scans. These images were useful for confirming that the GPU output was being generated, but they were not always useful for detecting meaningful anomalies. At low resolution they could lose structure, while at high resolution they often appeared as random-looking sparse dots.

The improved visual strategy focuses on **pattern-space** rather than only **number-space**.

Instead of asking:

```text
Where are the prime pixels?
```

the new visualisations ask:

```text
Which patterns are unusually productive?
Which patterns beat expected counts?
Which patterns are stable across independent ranges?
```

The most useful visual targets are now:

```text
actual / expected ratio
hits per million
singular-series score
pattern-space coordinates
range-by-range stability
```

This makes the graphics more directly connected to the current research goal: finding sustained A/E anomalies.

---

## Example build instructions

Example direct CUDA run:

```bash
./prime2 1000000000000 200000000 1 128 128
```

Arguments:

```text
./prime2 <start> <range_per_iteration> <iterations> <width> <height>
```

Example values:

```text
start:                1000000000000
range_per_iteration:  200000000
iterations:           1
image size:           128x128
```

---

## Example automated holdout search

```bash
python3 auto_explore.py \
  --chunks 30 \
  --start 2000000000000 \
  --max-zooms-per-chunk 2 \
  --max-depth 5 \
  --broad-width 128 \
  --broad-height 128 \
  --zoom-width 256 \
  --zoom-height 256
```

---

## Example exact-pattern validation

```bash
python3 pattern_prime_generator_v2.py \
  --pattern quad_0_6_126_336 \
  --start 1000000000000 \
  --limit 10000000 \
  --label test_1T_quad_0_6_126_336
```

---

## Example theory comparison

```bash
python3 theory_compare_patterns.py \
  --summary top4_10M_summary.csv \
  --prime-limit 200000
```

---

## Example A/E anomaly hunting

Create a pattern file:

```bash
cat > ae_patterns.txt <<'EOF'
quad_0_6_126_336
quad_0_6_96_210
quad_0_6_66_360
quad_0_6_120_336
quad_0_6_66_150
quad_0_6_156_300
quad_0_6_90_330
quad_0_6_126_270
quad_0_6_132_336
quad_0_6_66_90
quad_0_6_132_210
quad_0_6_90_120
quad_0_6_126_360
quad_0_6_126_246
quad_0_6_132_252
quad_0_6_84_90
EOF
```

Run the A/E hunter:

```bash
python3 ae_hunter.py \
  --patterns-file ae_patterns.txt \
  --starts 1000000000000,2000000000000,5000000000000,10000000000000 \
  --limit 10000000 \
  --threshold 1.2 \
  --results-csv ae_hunter_results.csv \
  --hits-csv ae_hunter_hits.csv
```

This tests each pattern across the selected ranges. If a pattern reaches:

```text
A/E >= 1.2
```

the result is saved to:

```text
ae_hunter_hits.csv
```

The script keeps going after finding a hit.

---

## Example long-running A/E search

To continue searching across many independent trillion-scale ranges:

```bash
python3 ae_hunter.py \
  --patterns-file ae_patterns.txt \
  --starts 1000000000000 \
  --iterations 50 \
  --start-step 1000000000000 \
  --limit 10000000 \
  --threshold 1.2 \
  --results-csv ae_hunter_long_results.csv \
  --hits-csv ae_hunter_long_hits.csv
```

This tests:

```text
1T
2T
3T
4T
...
50T
```

for each pattern in `ae_patterns.txt`.

---

## Example A/E visualisation

After running the hunter:

```bash
python3 ae_visuals.py \
  --results ae_hunter_results.csv \
  --prefix ae
```

Expected outputs:

```text
ae_top_patterns.csv
ae_scatter_ae_vs_singular.png
ae_timeline_top_patterns.png
ae_pattern_space_ae.png
ae_pattern_space_hits_per_million.png
```

These outputs are more useful than the early raw bitmap images because they show pattern-level structure rather than only sparse prime-position dots.

---

## Monitoring GPU usage

Show current GPU status:

```bash
/usr/bin/nvidia-smi
```

Live GPU monitor:

```bash
watch -n 1 "/usr/bin/nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv"
```

Check running project processes:

```bash
pgrep -af "prime2|auto_explore|python3"
```

Kill a stuck process owned by the current user:

```bash
kill PID_NUMBER
```

Force kill if needed:

```bash
kill -9 PID_NUMBER
```

---

## Running safely over SSH

For long-running jobs, use `tmux`:

```bash
tmux new -s prime_run
```

Start the run inside tmux, then detach:

```text
Ctrl-b
d
```

Reattach later:

```bash
tmux attach -t prime_run
```

This keeps the run alive if the SSH connection drops.

---

## What this project demonstrates

This project demonstrates the ability to:

* Build CUDA code for large-scale parallel computation.
* Process and analyse large numeric search spaces.
* Design a multi-stage computational experiment.
* Automate broad search, zooming, filtering, and validation.
* Develop custom statistical and theory-comparison tooling.
* Build an automatic anomaly-search loop.
* Create visualisations that are aligned with the research objective.
* Use empirical results to refine research direction.
* Avoid overclaiming by comparing results against theoretical expectations.

The strongest technical achievement is the complete pipeline:

```text
GPU scan → candidate detection → holdout validation → generator testing → theory comparison → A/E anomaly hunting → pattern-space visualisation

## Author
Nathan Shorter
