[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blessing-studentlens.streamlit.app/)

**[Open the hosted app](https://blessing-studentlens.streamlit.app/)**

# StudentLens — Study Habits & Student Outcomes
By Blessing Sekeramavi

An exploratory education research dashboard using the UCI Student Performance Portuguese-language dataset (649 records, 33 columns). Includes six research questions, descriptive distributions, bootstrap uncertainty, an adjusted association model and a leakage-aware prediction benchmark.

## Research questions
| Question | Variables | Visual |
|---|---|---|
| Do students who study more achieve higher final grades? | studytime, G3 | Box plot |
| How are absences associated with final grades? | absences, G3 | Scatter plot |
| Do grades differ by home internet access? | internet, G3 | Mean and 95% interval |
| How does parental education relate to achievement? | Medu, Fedu, G3 | Heatmap |
| Do higher-education plans align with different outcomes? | higher, G3 | Box plot |
| How do outcomes differ by educational support? | schoolsup, famsup, G3 | Mean and 95% interval |

## Run
Python 3.11+: `pip install -r requirements.txt`, then `python analyze.py`.
Preview: `python -m http.server 8000 --directory dist`, then open http://localhost:8000.
Alternatively run StudentLens.ipynb in Colab after uploading this project.
The dashboard requires no JavaScript packages or external CDNs. Deploy the dist directory using any static hosting provider (GitHub Pages supported).

## Design and methods
Source CSV remains unchanged (semicolon delimiter). Missing cells, exact duplicate rows and target ranges are checked. Zero grades are retained. No student ID is assumed. The two subject files are never concatenated; students may overlap across subjects.

Full-cohort means use 5,000 percentile bootstrap resamples, seed 42. School-filtered means use approximate t intervals; a singleton interval is not meaningful. Box plots use interpolated quartiles and 1.5 IQR whiskers. Heatmap suppresses cells with fewer than 10 records. Chart tables expose numeric values.

Adjusted OLS uses categorical study time, internet, parental education (linear ordinal codes), age, sex, school, prior failures, absences and higher-education plans. HC3 intervals do not address school clustering. Adjusted coefficients are conditional associations; ordinal linearity and residual model assumptions can be restrictive. R² is in-sample and not a prediction score. Comparisons are exploratory, without multiplicity correction or confirmatory p-value claims.

Ridge alpha=10 is fixed. Five shuffled KFold splits use seed 42. Scaling and categorical encoding occur within folds. Baseline predicts training-fold mean. G1, G2 and absences are excluded from the prediction inputs. Prior failures may already be known at enrolment, but feature availability must be assessed for any real deployment. MAE is reported per fold and averaged; random CV does not validate transfer to another school or future cohort. No individual deployment is proposed.

## Findings from this dataset
- Mean final grade: 11.91/20.
- Study-time group means: 10.84, 12.09, 13.23, 13.06; group sizes: 212, 305, 97, 35.
- Internet access group means: 12.17 (yes, n=498), 11.03 (no, n=151).
- Five-fold MAE: mean baseline 2.41; ridge 2.02.
These are historical sample results, not causal effects. The final study-time category does not continue a strictly increasing pattern.

## Limitations
Two schools; historical data; self-reported covariates; no sampling weights; nonrepresentative sample; possible confounding and reverse causation. Support recipients may start with greater needs. No programme effectiveness claim. Bootstrap/t intervals assume independent observations and omit clustering. No student satisfaction variable exists.

## Source and attribution
Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
https://archive.ics.uci.edu/dataset/320/student+performance
Dataset: CC BY 4.0. Original documentation is in dist/data/student.txt. Retain attribution. Project code: MIT (LICENSE).
