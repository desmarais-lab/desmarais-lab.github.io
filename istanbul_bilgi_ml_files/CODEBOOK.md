# Codebook & Application Guide

A guide to the ten reproduce-then-extend applications in this course. Each tutorial reproduces a published study's regression and then extends it with machine learning, evaluated out of sample. Open any notebook with the Colab badges in the [README](README.md).

## Overview

| # | Day | Application | Field | Method (regression → ML) |
|---|-----|-------------|-------|--------------------------|
| 01 | 1 | Voter turnout & term limits | Political science | OLS regression → out-of-sample prediction (train/test) |
| 02 | 1 | Civil-war onset (Fearon–Laitin 2003) | Political science / conflict | Logistic regression → out-of-sample ROC / AUC-PR |
| 05 | 2 | Militarized interstate disputes | Political science / conflict | Logistic RSF → RF / XGBoost (tuned) + SHAP; held-out test |
| 06 | 2 | Pinyon Jay habitat selection | Ecology | Logistic RSF → RF / XGBoost (tuned) + SHAP + partial dependence; held-out test |
| 07 | 2 | Snake occurrence / habitat | Ecology | Logistic RSF → RF / XGBoost (tuned) + SHAP + partial dependence; held-out test |
| 11 | 1 | Economic shocks & regional elite splits — Catalan Lliga (Vall-Prat 2022) | Comparative political economy | OLS (province FE) → lasso / ridge / elastic net + ABESS; repeated held-out (n=60) |
| 12 | 1 | Election-law reform after 2000 (Palazzolo & Moscardelli 2006) | American state politics | OLS → lasso / ridge / elastic net + ABESS; small-n (n=49) held-out |
| 13 | 1 | Determinants of perceived ageism (Du et al. 2025) | Gerontology / public health | OLS → LASSO selection + ridge/enet + ABESS; held-out (n=484) |
| 14 | 1 | Advance-care-planning engagement (Han et al. 2025) | Health services research | OLS → lasso/ridge/elastic net; held-out (n=276) |


---

## 01 — Voter turnout & term limits  (Day 1)

- **Field:** Political science
- **Method:** OLS regression → out-of-sample prediction (train/test)
- **Data file:** [`turnout_state.tab`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/turnout_state.tab)
- **Notebook:** [`notebooks/01_turnout_day1.ipynb`](notebooks/01_turnout_day1.ipynb)
- **Published study:** Kuhlmann, R. & Lewis, D. C. (2017). "Legislative Term Limits and Voter Turnout." *State Politics & Policy Quarterly* 17(4):372–392. Data: UNC Dataverse doi:10.15139/S3/POGNLC. The original analysis is an **OLS** regression of state lower-chamber turnout on institutional predictors (Table 2, Col. 1). This example illustrates **out-of-sample prediction with a regression**.

**Unit of analysis:** a U.S. state lower-chamber legislative election (a **state-year**); n = 486.

| Variable | Definition |
|---|---|
| `turnout` | lower-chamber election turnout rate, % of the voting-age population **(outcome)** |
| `tlness` | term-limitedness: strength/bindingness of legislative term limits |
| `lcr2` | log legislator-to-citizen ratio (per 100,000 residents) |
| `mmds` | share of legislative seats elected from multimember districts |
| `regdifficulty` | voter-registration closing date (days before the election) |
| `straight` | 1 if the state offers straight-party-ticket voting |
| `iuse2` | log number of ballot initiatives on the ballot |
| `squire` | Squire index of legislative professionalism |
| `ranney` | Ranney index of interparty electoral competition |
| `income2` | per-capita income (thousands of dollars) |
| `diversity3` | Herfindahl index of racial/ethnic diversity |
| `preselec` | 1 in presidential-election years |
| `redistrict` | 1 if districts were newly redrawn (post-redistricting) |
| `year` | election year |

---

## 02 — Civil-war onset (Fearon–Laitin 2003)  (Day 2)

- **Field:** Political science / conflict
- **Method:** Logistic regression → SVM / k-NN / random forest; SHAP interpretation
- **Data file:** [`SambnisImp.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/SambnisImp.csv)
- **Notebook:** [`notebooks/02_fearon_laitin_day2.ipynb`](notebooks/02_fearon_laitin_day2.ipynb)
- **Published study:** Fearon, J. D. & Laitin, D. D. (2003). "Ethnicity, Insurgency, and Civil War." *American Political Science Review* 97(1):75–90. A **logistic regression** of civil-war onset on 11 country-year predictors (Sambanis 2006 imputed data; Harvard Dataverse doi:10.7910/DVN/KRKWK8). Onset is a **rare event** (~1.6%). The tutorial reproduces the logit, then extends it with **SVM, k-NN, and random forests** (honest AUC-PR on a held-out test set) and interprets the drivers with **SHAP**.

**Unit of analysis:** country-year; n = 7,140 (116 civil-war onsets, ~1.6%).

| Variable | Definition |
|---|---|
| `warstds` | civil-war onset: 1 in the first year of a new civil war **(outcome)** |
| `warhist` | prior civil-war history in the country |
| `ln_gdpen` | log GDP per capita (lagged one year) |
| `lpopns` | log national population |
| `lmtnest` | log % of the country that is mountainous terrain |
| `ncontig` | 1 if the state has noncontiguous territory |
| `oil` | 1 if fuel exports exceed one-third of total exports (oil exporter) |
| `nwstate` | 1 in the first two years of a newly independent state |
| `inst3` | political instability: a >=3-point Polity change in the prior 3 years |
| `pol4` | Polity score (-10 autocracy to +10 democracy) |
| `ef` | ethnolinguistic fractionalization |
| `relfrac` | religious fractionalization |

---

## 05 — Militarized interstate disputes  (Day 2)

- **Field:** Political science / conflict
- **Method:** Logistic RSF → RF / XGBoost (tuned) + SHAP; held-out test
- **Data file:** [`gb20.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/gb20.csv)
- **Notebook:** [`notebooks/05_gibler_braithwaite_day2.ipynb`](notebooks/05_gibler_braithwaite_day2.ipynb)
- **Published study:** Gibler, D. M. & Braithwaite, A. (2013). "Dangerous Neighbours, Regional Territorial Conflict and the Democratic Peace," *British Journal of Political Science* 43(4):877 (doi:10.1017/S000712341200052X); replication data: Harvard Dataverse doi:10.7910/DVN/HQLOOJ) — a binary-response regression for whether a dyadic militarized dispute becomes **fatal**, on dyadic predictors. This is *interstate* conflict, distinct from the civil-war cases; fatal disputes are rare (~1.3%) → scored by **AUC-PR**.

**Unit of analysis:** a militarized interstate dispute-dyad; n = 20,263 (~1.3% fatal).

| Variable | Definition |
|---|---|
| `fatal` | 1 if the militarized dispute produced at least one battle death **(outcome)** |
| `lowdem` | lower of the two states' Polity democracy scores (weakest-link) |
| `terr500k10yr` | 1 if the dispute is over territory |
| `defense` | 1 if the two states share a defensive alliance |
| `caprat` | national-capability ratio (stronger to weaker state, CINC) |
| `contig` | 1 if the states are geographically contiguous |
| `peaceyears` | years since the dyad's previous dispute |
| `spline1`-`spline3` | cubic-spline bases for peace-years (temporal dependence) |
| `interaction4`, `pnbdem_low` | constructed interaction terms from the published specification |

---

## 06 — Pinyon Jay habitat selection  (Day 2)

- **Field:** Ecology
- **Method:** Logistic RSF → RF / XGBoost (tuned) + SHAP + partial dependence; held-out test
- **Data file:** [`jays_clean.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/jays_clean.csv)
- **Notebook:** [`notebooks/06_pinyonjay_rsf_day2.ipynb`](notebooks/06_pinyonjay_rsf_day2.ipynb)
- **Published study:** Boone, J. D., Witt, C. & Ammon, E. M. (2021). "Behavior-specific occurrence patterns of Pinyon Jays." *PLoS ONE* (doi:10.1371/journal.pone.0237621; S1 Dataset, open). A **logistic resource-selection function (RSF)** — habitat *use* points (1) vs *available* points (0) on environmental predictors. Use is the minority class, so out-of-sample skill is scored by **AUC-PR** (area under the precision-recall curve).

**Unit of analysis:** a sampled ground location -- an observed jay-**use** point vs an **available**
(background) point; n = 466 (124 use).

| Variable | Definition |
|---|---|
| `use` | 1 = observed Pinyon Jay use location, 0 = available location **(outcome)** |
| `Elevation..feet.` | elevation (feet) |
| `Slope..percent.` | ground slope (%) |
| `Stand_Age..years.` | forest stand age (years) |
| `Stand_Density_Index` | stand density index (tree density / competition) |
| `Canopy_Cover..percent.` | overstory canopy cover (%) |
| `Tree_Cover..percent.` | tree cover (%) |
| `Shrub_Cover..percent.` | shrub cover (%) |
| `Forb_Cover..percent.` | forb (herbaceous) cover (%) |
| `Grass_Cover..percent.` | grass cover (%) |
| `Woody_Debris..category.1.`-`.3.` | counts of downed woody debris by size class (1-3) |
| `Distance.to.Road..km.` | distance to the nearest road (km) |
| `DisturbanceYes` | 1 if any recent disturbance was recorded at the site |
| `Distance.to.Edge..m.` | distance to the nearest habitat edge (m) |

---

## 07 — Snake occurrence / habitat  (Day 2)

- **Field:** Ecology
- **Method:** Logistic RSF → RF / XGBoost (tuned) + SHAP + partial dependence; held-out test
- **Data file:** [`snake_clean.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/snake_clean.csv)
- **Notebook:** [`notebooks/07_snake_rsf_day2.ipynb`](notebooks/07_snake_rsf_day2.ipynb)
- **Published study:** Fill, J. M., Waldron, J. L., Welch, S. M. & Gibbons, J. W. (2015). "Using Multiscale Spatial Models to Assess Potential Surrogate Habitat for an Imperiled Reptile," *PLoS ONE* (doi:10.1371/journal.pone.0123307; S1 Dataset, open). A **logistic resource-selection function** — *used* (1) vs *available* (0) locations (n = 2075) on canopy cover, shrub cover, aspect, elevation and ground cover. Use-vs-availability is the ecology workhorse design → scored by **AUC-PR**.

**Unit of analysis:** a sampled location -- a telemetry-**used** location vs an **available** location; n = 2,075.

| Variable | Definition |
|---|---|
| `Response` | 1 = used location, 0 = available location **(outcome)** |
| `CanCov` | canopy cover (%) |
| `ShrubCov` | shrub cover (%) |
| `Aspect` | slope aspect (compass orientation / index) |
| `Elevation` | elevation |
| `Cover.` | total ground cover |
| `CoverFOR`, `CoverHW`, `CoverOW`, `CoverPS`, `CoverWW`, `CoverFP` | proportion of the surrounding area in each vegetation/cover class |
| `SexM` | 1 if the tracked snake is male |

---




## 11 — Economic shocks & regional elite splits (Catalan Lliga)  (Day 1)

- **Field:** Comparative political economy / historical political economy
- **Method:** OLS (province fixed effects) → lasso / ridge / elastic net + ABESS best subset; repeated held-out (n = 60)
- **Data file:** [`catalan_lliga.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/catalan_lliga.csv)
- **Notebook:** [`notebooks/11_economic_shocks_lliga_day1.ipynb`](notebooks/11_economic_shocks_lliga_day1.ipynb)
- **Published study:** Vall-Prat, P. (2022). "Economic Shocks, Mobilization, and Regional Elite Splits." *Comparative Political Studies* 56(2) (doi:10.1177/00104140221089641). Replication data: Harvard Dataverse doi:10.7910/DVN/NYS9UR. A cross-sectional OLS of Catalan Lliga electoral success on the 1898 colonial trade shock + controls, with province fixed effects — 60 districts, a small-sample design where **regularization improves out-of-sample prediction and selects the drivers**.

**Unit of analysis:** Catalan electoral district; n = 60. Outcome `Lliga` = number of Lliga wins 1901–1918; key predictor `ColShock` = exposure to the 1898 colonial trade shock; controls: identity mobilization, log population, distance to Manresa, landowner dominance, Barcelona; province fixed effects.


## 12 — Election-law reform after the 2000 election  (Day 1)

- **Field:** American state politics
- **Method:** OLS → lasso / ridge / elastic net + ABESS best subset; repeated held-out (n = 49)
- **Data file:** [`election_law_reform.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/election_law_reform.csv)
- **Notebook:** [`notebooks/12_election_law_reform_day1.ipynb`](notebooks/12_election_law_reform_day1.ipynb)
- **Published study:** Palazzolo, D. J. & Moscardelli, V. G. (2006). "Policy Crisis and Political Leadership: Election Law Reform in the States after the 2000 Presidential Election." *State Politics & Policy Quarterly* 6(3):300–321 (doi:10.1177/153244000600600303). Replication data: UNC SPPQ Dataverse doi:10.15139/S3/12145. A cross-sectional OLS of states' weighted reform count on 13 predictors (49 states, Florida excluded) — a high predictor-to-observation ratio that illustrates **regularization taming over-fitting and selecting the drivers** in a small-n design.

**Unit of analysis:** U.S. state; n = 49 (Florida excluded as the 2000 outlier).

| Variable | Definition |
|---|---|
| `reformwt` | weighted count of election reforms adopted (outcome) |
| `leadindx` | leadership index |
| `ranney` | party competitiveness (folded Ranney) |
| `simple` | simple divided government |
| `compound` | compound divided government |
| `termrank` | legislative term limits |
| `culture` | political culture |
| `ideology` | conservative state ideology (Norrander) |
| `fiscalin` | ratio of state revenues to expenditures |
| `lwv_act` | interest-group mobilization (League of Women Voters) |
| `ffactor` | winner's margin in the 2000 presidential election |
| `residvot` | residual (uncounted) vote rate in 2000 |
| `ffrxrv` | interaction of 2000 margin (recoded) and residual vote |
| `commrec` | number of reform-commission recommendations |

## 13 — Determinants of perceived ageism (Du et al. 2025)  (Day 1)

- **Field:** Gerontology / public health
- **Method:** OLS → LASSO variable selection (as the authors used) + ridge / elastic net + ABESS; repeated held-out (n = 484)
- **Data file:** [`perceived_ageism.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/perceived_ageism.csv)
- **Notebook:** [`notebooks/13_perceived_ageism_day1.ipynb`](notebooks/13_perceived_ageism_day1.ipynb)
- **Published study:** Du, J., Wang, M. & Wu, X. (2025). "The current status and determinants of perceived ageism among community-dwelling older adults." *PLOS ONE* 20 (doi:10.1371/journal.pone.0330254; S1 Dataset). Multiple regression of perceived ageism on 8 LASSO-selected predictors (n=484); reproduces the reported determinants (education, vision, hearing) and the authors' LASSO selection.

## 14 — Advance-care-planning engagement (Han et al. 2025)  (Day 1)

- **Field:** Health services research
- **Method:** OLS → lasso / ridge / elastic net; repeated held-out (n = 276)
- **Data file:** [`surrogate_acp.csv`](https://raw.githubusercontent.com/desmarais-lab/desmarais-lab.github.io/master/istanbul_bilgi_ml_files/data/surrogate_acp.csv)
- **Notebook:** [`notebooks/14_surrogate_acp_day1.ipynb`](notebooks/14_surrogate_acp_day1.ipynb)
- **Published study:** Han, Z., Su, J. & Ma, G. (2025). "Factors influencing the participation of surrogate decision-makers for advance care planning." *PLOS ONE* 20 (doi:10.1371/journal.pone.0325551; S1 Dataset). Multiple regression of surrogate ACP-engagement on personal/clinical factors (n=276); reproduces the reported positive predictors (decision experience, education, ACP knowledge, treatment expenditure).
