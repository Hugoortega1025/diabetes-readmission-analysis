
<div align="center">
  <h1>Clinical Risk Audit: Mitigating HRRP Financial Penalties</h1>
  <p><strong>Healthcare Analytics &nbsp;|&nbsp; 101,766 Patient Encounters &nbsp;|&nbsp; SQL-Driven Risk Analysis</strong></p>
</div>

## Background
Under the Hospital Readmissions Reduction Program (HRRP), Hospitals face direct finanacial penalties for execissive 30-day readmissions. For diabetic patients — a population already at elevated risk for serious long-term complications — an early readmission is not just a cost event. It signals a failure somewhere in the discharge process: rushed planning, poor follow-up coordination, or a clinical picture that was more complex than the encounter data captured.

The Objective: Identify high-risk "Penalty Clusters" before discharge so the hospitals can deploy targeted interventions (follow-ups, pharmacist reviews) to improve patient outcomes but also avoid financial penalty events.


## Executive Summary
Across 130 US hospitals, 11% (2008) of diabetic patient encounters result in early readmission — an overall rate that HRRP penalizes hospitals for exceeding. The dominant driver is not diagnosis complexity but a patient's history of prior hospital admissions.

Readmission rates climb from 8% for patients with no prior admissions to 44% for patients with 8 or more — a 5.5x increase, consistent across every subgroup tested. Notably, this same driver fully explains an initially puzzling signal: young adults (20-30) showed the second-highest readmission rate of any age group, but breaking that group down by prior admission history shows they follow the identical pattern as every other age band — there is no separate, unexplained young-adult risk factor in this dataset.

A single actionable rule — flag any patient with 2+ prior inpatient visits — requires no predictive model or new infrastructure and identifies the population driving the majority of HRRP-countable readmissions.




## Business Problem
Clinical staff operating under heavy caseloads face a consistent problem: not every patient who needs enhanced discharge planning receives it, leading to patients having to be readmitted. The financial stakes are direct. Under HRRP, hospitals with excessive readmission rates for high-risk populations including diabetic patients face Medicare payment reductions. Beyond the penalty, each early readmission represents additional resource consumption, extended staff burden, and a worsened patient outcome.

The aim of this analysis: use encounter-level data already collected at every admission to identify high-risk Penalty Clusters before discharge — giving discharge planning teams recommendations they can act on.

----


### Age Distribution

![age_readmission](visuals/age_readmission.png)

Young adults (20-30) showed a surprisingly elevated 14% early readmission rate — higher than every age band except patients over 60. This raised the question addressed below: is age itself a risk factor, or is something else driving this?


### The Penalty Cluster — Steady Insulin, Multiple Prior Admissions

| Attribute | Value |
|---|---|
| Age band | 50–80 |
| Prior inpatient visits |3-5 |
| Medication status | Insulin-dependent, steady dose |
| Readmission rate | 22.6–23.3% |

This is the clearest, most consistent Penalty Cluster in the dataset — the rate holds within a single percentage point across three separate 10-year age bands once insulin status and admission history are fixed, meaning these two factors drive risk more than age does. High prior admission counts signal chronic instability; a steady insulin dose without escalation suggests the current regimen may not be adequately controlling the condition between encounters.

![Top 10 High-Risk Readmission Profiles](visuals/cte_high_risk_profiles.png)


### Resolving the Young-Adult Question

The age-group elevation seen above is not a separate, unexplained risk factor — it's the same driver at work in a smaller population. Breaking young adults down by prior inpatient visits shows an identical escalating pattern to the rest of the dataset:

| Prior Inpatient Visits | Readmission Rate |
|---|---|
| 0 visits | 5.5% |
| 1-2 visits | 20.9% |
| 3+ visits | 41.6% |

![Young Adult Early Readmission Rate by Prior Inpatient Visits](visuals/young_adult_inpatient_readmission.png)

Young adults with no admission history are actually below the hospital's overall average. The elevated group-wide rate is driven entirely by the smaller subset with repeat admissions — the same signal that explains risk everywhere else in this population. There is no distinct young-adult risk category requiring separate data collection; the existing signal already covers it.

## Key Finding — Prior Inpatient Visits Drive Everything

SQL exploration phase, conducted in Databricks, consistently surfaced one variable above all others: prior inpatient visits.

| Prior Inpatient Visits | Readmission Rate |
|---|---|
| 0 | 8.4% |
| 1–2 | 14.2% |
| 3–5 | 22.8% |
| 6–7 | 34.9% |
| 8+ | 43.3% |

![Early Readmission Rate by Prior Inpatient Visits](visuals/inpatient_visits_readmission.png)

This relationship was confirmed independently by SHAP analysis on the trained Random Forest — prior inpatient visits ranked as the dominant feature contribution to individual readmission predictions, ahead of all medication, diagnosis, and demographic variables. It also holds true within the young-adult subgroup above, confirming it as a universal driver rather than one specific to older patients.

**Business implication:** A patient's readmission history is both the most accessible and most predictive variable in the dataset. Hospitals already have this information at the point of discharge. The gap is not data — it is a systematic process for acting on it.



## Recommendation
**Stakeholder:** Hospital discharge planning teams and clinical operations leadership

Flag all patients with 2+ prior inpatient visits for enhanced discharge planning — regardless of age. This single rule captures the highest-risk population across every age group in the dataset and requires no predictive model or new infrastructure to implement. Pair this with closer review for patients aged 50-80 on a steady insulin dose, the sharpest single Penalty Cluster identified.



## Methodology
SQL Exploration (Databricks) — 10-step EDA using CTEs, GROUP BY aggregations, CASE WHEN logic, HAVING filters, and subqueries across 101,766 patient encounters to identify readmission patterns by age, race, medications, diagnoses, insulin usage, and prior inpatient visits.




## Next Steps & Limitations
**Data limitation:** This dataset only captures readmissions within the same health system — a patient readmitted to a different hospital would not be recorded here. The 11% (2008) rate found in this analysis is likely an underestimate relative to broader claims-based estimates.

**Scope limitation:** Social determinants of health, medication adherence records, mental health history, and post-discharge follow-up data are absent from this dataset entirely — these are plausible contributors to risk that a purely clinical dataset cannot capture.

**Next step:** Validate the 2+ prior visits rule against a more recent, cross-network dataset (e.g. claims-based data) to confirm it holds outside this single-network sample.

---


## Tools & Platform
- **Platform:** Databricks
- **Languages:** SQL

## Project Structure
- `clinical_diabetes_EDA.ipynb` — SQL EDA (Databricks)


