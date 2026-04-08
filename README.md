# Diabetes Hospital Readmission Analysis

## Overview
SQL and Python analysis of 101,766 diabetic patient encounters across 130 US hospitals (1999-2008) to identify key clinical indicators of early readmission within 30 days.

## Problem Statement
11% of diabetic patients in this dataset are readmitted within 30 days of discharge. Using clinical data available given the dataset, this project aims to see if a risk flagging system can identify high risk patients before they are discharged.

## Key Findings
- Prior inpatient visits are the strongest predictor of early readmission (8% → 44%)
- Older patients (60-80) with multiple prior admissions on insulin show 20-28% readmission rates
- Young adults (20-30) show elevated rates unexplained by clinical features alone
- Model achieves 64% AUC — below performance for reliable clinical deployment. 

## Tools & Platform
- **Platform:** Databricks
- **Languages:** SQL, Python (Pandas, Scikit-learn, SHAP)

## Project Structure
- `diabetes_readmission_analysis (1).ipynb` — SQL EDA (Databricks)
- `diabetes_readmission_modeling.ipynb` — Statistical testing, ML modeling, SHAP, Risk flagging (Databricks)
