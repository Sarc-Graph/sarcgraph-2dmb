# Metrics Guide 🔬

This document provides a comprehensive explanation of the physiological metrics extracted by the analysis pipeline, detailing what each metric represents and how it is computed.

---

## 1. Cycle-Based Metrics

These metrics describe the overall rhythmic activity of the tissue.

* **`num_contractions`**

    * **What it represents and how it's computed:** The total number of complete contraction cycles (relaxed $\rightarrow$ contraction $\rightarrow$ relaxation) successfully identified within the analyzed signal. This is determined by first computing the average sarcomere length signal, denoising it using Gaussian Process Regression (GPR), and then using the `analyze.py` module to detect the cell's state and count the number of complete cycles.

* **`contraction_period`**

    * **What it represents:** The average duration of one complete contraction cycle (in seconds, based on `FRAME_RATE`). This is a measure of how long it takes for the tissue to go through one full beat.

* **`contraction_frequency`**

    * **What it represents:** The average number of contraction cycles per unit of time (e.g., beats per second). This is the inverse of the contraction period.

---

## 2. Sarcomere Length Statistics

These metrics describe the average sarcomere length during specific states. All lengths are in **microns ($\mu m$)**.

* **`relaxed_sarcomere_length_mean`**

    * **What it represents and how it's computed:** The average sarcomere length when the tissue is in its most relaxed state (i.e., at peak lengthening). This is computed by first calculating the mean sarcomere length for each individual frame, and then averaging these per-frame mean lengths across all frames identified as part of a "relaxed state" by the `analyze.py` module.

* **`relaxed_sarcomere_length_median`**

    * **What it represents and how it's computed:** The average of the median sarcomere lengths computed for each individual frame across all frames identified as part of a "relaxed state".

* **`relaxed_sarcomere_length_q25`**

    * **What it represents and how it's computed:** The average of the 25th percentile (first quartile) of sarcomere lengths computed for each individual frame across all frames identified as part of a "relaxed state".

* **`relaxed_sarcomere_length_q75`**

    * **What it represents and how it's computed:** The average of the 75th percentile (third quartile) of sarcomere lengths computed for each individual frame across all frames identified as part of a "relaxed state".

* **`relaxed_sarcomere_length_std`**

    * **What it represents and how it's computed:** The average of the standard deviation of sarcomere lengths computed for each individual frame across all frames identified as part of a "relaxed state", indicating variability within those frames.

* **`peak_sarcomere_length_mean`**

    * **What it represents and how it's computed:** The average sarcomere length when the tissue is at its most contracted state (i.e., at peak shortening). This is computed by first calculating the mean sarcomere length for each individual frame, and then averaging these per-frame mean lengths across a small window of frames (defined by `FRAME_AVG_RANGE`) around the identified peak contraction frame for each cycle.

* **`peak_sarcomere_length_median`**

    * **What it represents and how it's computed:** The average of the median sarcomere lengths computed for each individual frame across a small window of frames (defined by `FRAME_AVG_RANGE`) around the identified peak contraction frame for each cycle.

* **`peak_sarcomere_length_q25`**

    * **What it represents and how it's computed:** The average of the 25th percentile of sarcomere lengths computed for each individual frame across a small window of frames (defined by `FRAME_AVG_RANGE`) around the identified peak contraction frame for each cycle.

* **`peak_sarcomere_length_q75`**

    * **What it represents and how it's computed:** The average of the 75th percentile of sarcomere lengths computed for each individual frame across a small window of frames (defined by `FRAME_AVG_RANGE`) around the identified peak contraction frame for each cycle.

* **`peak_sarcomere_length_std`**

    * **What it represents and how it's computed:** The average of the standard deviation of sarcomere lengths computed for each individual frame across a small window of frames (defined by `FRAME_AVG_RANGE`) around the identified peak contraction frame for each cycle, indicating variability within those frames.

---

## 3. Amplitude and Velocity Metrics

These metrics quantify the magnitude and speed of sarcomere changes.

* **`shortening_amplitude`**

    * **What it represents:** The average maximum change in sarcomere length from its relaxed state to its peak contracted state during a cycle. This is a key measure of contractility. The final metric is the mean of these per-cycle amplitudes.

* **`peak_shortening_velocity`**

    * **What it represents:** The fastest rate at which sarcomeres shorten during a contraction phase. A more negative value indicates faster shortening. Units are $\mu m/s$.

    * **How it's computed:** Calculated as the minimum (most negative) value of the time-based derivative of the GPR-smoothed sarcomere length signal during each identified contraction phase. The final metric is the average of these peak velocities across cycles.

* **`peak_lengthening_velocity`**

    * **What it represents:** The fastest rate at which sarcomeres lengthen during a relaxation (expansion) phase. A more positive value indicates faster lengthening. Units are $\mu m/s$.

    * **How it's computed:** Calculated as the maximum (most positive) value of the time-based derivative of the GPR-smoothed sarcomere length signal during each identified relaxation (expansion) phase. The final metric is the average of these peak velocities across cycles.

---

## 4. Timing Metrics

These metrics describe the duration of different phases within the contraction-relaxation cycle. All times are in **seconds (s)**.

* **`contraction_onset_to_relaxation_end_time`**

    * **What it represents:** The total duration from the beginning of a contraction phase to the end of the subsequent relaxation phase. This is the duration of the active part of the cycle.

* **`contraction_onset_to_peak_contraction_time`**

    * **What it represents:** The time taken from the beginning of a contraction to reach its peak (maximum shortening).

* **`contraction_onset_to_50_contracted_time`**

    * **What it represents:** The time taken from the beginning of a contraction to reach 50% of its total shortening amplitude.

* **`half_contracted_to_peak_contraction_time`**

    * **What it represents:** The time taken to go from 50% of total shortening to the peak (full) contraction.

* **`peak_contraction_to_relaxation_end_time`**

    * **What it represents:** The time taken from the peak of contraction to the end of the relaxation phase.

* **`peak_contraction_to_50_relaxed_time`**

    * **What it represents:** The time taken from the peak of contraction to reach 50% of its total relaxation (lengthening) amplitude.

* **`half_relaxed_to_full_relaxation_time`**

    * **What it represents:** The time taken to go from 50% relaxation to full relaxation.

---

## 5. Tissue Geometry Metrics

These metrics describe the physical dimensions and area of the tissue. All lengths are in **microns ($\mu m$)** and areas in **square microns ($\mu m^2$)**.

* **`tissue_length_relaxed`**

    * **What it represents:** The average length of the tissue's bounding box during relaxed states.

* **`tissue_length_peak_contraction`**

    * **What it represents:** The average length of the tissue's bounding box during peak contraction states. For averaging, a small window of frames (defined by `FRAME_AVG_RANGE`) around the peak contraction are considered and averaged.

* **`tissue_width_relaxed`**

    * **What it represents:** The average width of the tissue's bounding box during relaxed states. 

* **`tissue_width_peak_contraction`**

    * **What it represents:** The average width of the tissue's bounding box during peak contraction states. For averaging, a small window of frames (defined by `FRAME_AVG_RANGE`) around the peak contraction are considered and averaged.

* **`tissue_area_convex_hull_relaxed`**

    * **What it represents:** The average area of the convex hull (a tight-fitting polygon around the tissue) during relaxed states.

* **`tissue_area_convex_hull_peak_contraction`**

    * **What it represents:** The average area of the convex hull during peak contraction states. For averaging, a small window of frames (defined by `FRAME_AVG_RANGE`) around the peak contraction are considered and averaged.

* **`tissue_area_bounding_box_relaxed`**

    * **What it represents:** The average area of the rectangular bounding box enclosing the tissue during relaxed states (basically length $\times$ width).

* **`tissue_area_bounding_box_peak_contraction`**

    * **What it represents:** The average area of the rectangular bounding box enclosing the tissue during peak contraction states (basically length $\times$ width). For averaging, a small window of frames (defined by `FRAME_AVG_RANGE`) around the peak contraction are considered and averaged.

---

## 6. Sarcomere Count and Density Metrics

These metrics relate the number of detected sarcomeres to the tissue area.

* **`total_number_of_sarcomeres`**

    * **What it represents:** The average number of individual sarcomeres detected per frame across the entire video.

* **`sarcomere_density_convex_hull`**

    * **What it represents:** The average density of sarcomeres within the convex hull area of the tissue. Units are sarcomeres/$\mu m^2$.

    * **How it's computed:** Mean of (number of sarcomeres per frame / convex hull area per frame).

* **`sarcomere_density_bounding_box`**

    * **What it represents:** The average density of sarcomeres within the bounding box area of the tissue. Units are sarcomeres/$\mu m^2$.

    * **How it's computed:** Mean of (number of sarcomeres per frame / bounding box area per frame).

---

## 7. Noise Metric

This metric quantifies the effectiveness of the denoising step.

* **`noise_level`**

    * **What it represents:** A measure of the residual noise in the signal after Gaussian Process Regression (GPR) smoothing. A lower value indicates more effective smoothing.

    * **How it's computed:** The standard deviation of the difference between the raw mean sarcomere length signal and the GPR-smoothed sarcomere length signal.