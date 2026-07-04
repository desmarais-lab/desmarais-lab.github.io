# Codebook & Application Guide

A guide to the ten reproduce-then-extend applications in this course. Each tutorial reproduces a published study's regression and then extends it with machine learning, evaluated out of sample. Open any notebook with the Colab badges in the [README](README.md).

## Overview

| # | Day | Application | Field | Method (regression → ML) |
|---|-----|-------------|-------|--------------------------|
| 01 | 1 | Voter turnout & term limits | Political science | OLS regression → out-of-sample prediction (train/test) |
| 02 | 1 | Civil-war onset (Fearon–Laitin 2003) | Political science / conflict | Logistic regression → out-of-sample ROC / AUC-PR |
| 03 | 1 | Civil-war onset (Collier–Hoeffler 2004) | Political economy / conflict | Logistic regression → lasso / ridge / elastic net (1-SE) + ABESS best subset |
| 04 | 1 | Civil-war onset (Hegre–Sambanis 2006) | Political science / conflict | Logistic regression → lasso / ridge / elastic net (1-SE) + ABESS best subset |
| 05 | 2 | Militarized interstate disputes | Political science / conflict | Logistic RSF → RF / XGBoost (tuned) + SHAP; held-out test |
| 06 | 2 | Pinyon Jay habitat selection | Ecology | Logistic RSF → RF / XGBoost (tuned) + SHAP + partial dependence; held-out test |
| 07 | 2 | Snake occurrence / habitat | Ecology | Logistic RSF → RF / XGBoost (tuned) + SHAP + partial dependence; held-out test |
| 08 | 1 | Ames house prices | Real estate / hedonic economics | OLS hedonic regression → lasso / ridge / elastic net (1-SE) + ABESS; held-out test |
| 09 | 1 | Building energy efficiency | Engineering | Linear regression → polynomial regression (CV-selected degree); held-out test |
| 10 | 1 | Wine quality | Food science / chemometrics | Multiple regression → polynomial regression (CV-selected degree); held-out test |


---

## 01 — Voter turnout & term limits  (Day 1)

- **Field:** Political science
- **Method:** OLS regression → out-of-sample prediction (train/test)
- **Data file:** [`turnout_state.tab`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/turnout_state.tab)
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

## 02 — Civil-war onset (Fearon–Laitin 2003)  (Day 1)

- **Field:** Political science / conflict
- **Method:** Logistic regression → out-of-sample ROC / AUC-PR
- **Data file:** [`SambnisImp.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/SambnisImp.csv)
- **Notebook:** [`notebooks/02_fearon_laitin_day1.ipynb`](notebooks/02_fearon_laitin_day1.ipynb)
- **Published study:** Fearon, J. D. & Laitin, D. D. (2003). "Ethnicity, Insurgency, and Civil War." *American Political Science Review* 97(1):75–90. A **logistic regression** of civil-war onset on 11 country-year predictors (Sambanis 2006 imputed data; Harvard Dataverse doi:10.7910/DVN/KRKWK8). Onset is a **rare event** (~1.6%); this example illustrates **out-of-sample classification** and the right evaluation metric for imbalanced data.

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

## 03 — Civil-war onset (Collier–Hoeffler 2004)  (Day 1)

- **Field:** Political economy / conflict
- **Method:** Logistic regression → lasso / ridge / elastic net (1-SE) + ABESS best subset
- **Data file:** [`SambnisImp.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/SambnisImp.csv)
- **Notebook:** [`notebooks/03_collier_hoeffler_day1.ipynb`](notebooks/03_collier_hoeffler_day1.ipynb)
- **Published study:** Collier, P. & Hoeffler, A. (2004). "Greed and Grievance in Civil War." *Oxford Economic Papers* 56(4):563–595 — the "greed vs grievance" **logistic regression** of civil-war onset on 12 economic/structural predictors (Sambanis 2006 data; Harvard Dataverse doi:10.7910/DVN/KRKWK8). This example illustrates **regularization and variable selection**.

**Unit of analysis:** country-year; onset base rate ~1.6%.

| Variable | Definition |
|---|---|
| `warstds` | civil-war onset **(outcome)** |
| `sxpnew` | primary-commodity exports as a share of GDP |
| `sxpsq` | square of `sxpnew` (captures the inverted-U in commodity dependence) |
| `ln_gdpen` | log GDP per capita |
| `gdpgrowth` | annual GDP growth rate |
| `warhist` | prior civil-war history |
| `lmtnest` | log % mountainous terrain |
| `ef` | ethnolinguistic fractionalization |
| `popdense` | population density |
| `lpopns` | log population |
| `coldwar` | 1 during the Cold War period |
| `seceduc` | secondary-school enrollment (male) |
| `ptime` | time at peace (years) since the previous war |

---

## 04 — Civil-war onset (Hegre–Sambanis 2006)  (Day 1)

- **Field:** Political science / conflict
- **Method:** Logistic regression → lasso / ridge / elastic net (1-SE) + ABESS best subset
- **Data file:** [`SambnisImp.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/SambnisImp.csv)
- **Notebook:** [`notebooks/04_hegre_sambanis_day1.ipynb`](notebooks/04_hegre_sambanis_day1.ipynb)
- **Published study:** Hegre, H. & Sambanis, N. (2006). "Sensitivity Analysis of Empirical Results on Civil War Onset." *Journal of Conflict Resolution* 50(4):508–535 — a **logistic regression** built from the predictors their sensitivity analysis found most robust (Sambanis 2006 data; Harvard Dataverse doi:10.7910/DVN/KRKWK8). This example illustrates **regularization and variable selection**.

**Unit of analysis:** country-year; onset base rate ~1.6%.

| Variable | Definition |
|---|---|
| `warstds` | civil-war onset **(outcome)** |
| `lpopns` | log population |
| `ln_gdpen` | log GDP per capita |
| `inst3` | political instability (recent regime change) |
| `parreg` | Polity regulation-of-participation score |
| `geo34`, `geo1` | geographic-region indicators |
| `proxregc` | proximity to a regional/neighboring conflict |
| `gdpgrowth` | GDP growth rate |
| `anoc` | anocracy: 1 for semi-democracies (Polity -5 to +5) |
| `partfree` | 1 if 'partly free' (Freedom House) |
| `nat_war` | 1 if a neighboring state is at war |
| `lmtnest` | log % mountainous terrain |
| `decade1` | decade indicator |
| `pol4sq` | Polity score squared |
| `nwstate` | newly independent state (first two years) |
| `regd4_alt` | regime durability (years since the last regime change) |
| `etdo4590` | ethnic dominance: largest group 45-90% of the population |
| `milper` | military personnel (size of the armed forces) |
| `tnatwar` | number of neighboring states at war |
| `presi` | 1 if a presidential political system |

---

## 05 — Militarized interstate disputes  (Day 2)

- **Field:** Political science / conflict
- **Method:** Logistic RSF → RF / XGBoost (tuned) + SHAP; held-out test
- **Data file:** [`gb20.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/gb20.csv)
- **Notebook:** [`notebooks/05_gibler_braithwaite_day2.ipynb`](notebooks/05_gibler_braithwaite_day2.ipynb)
- **Published study:** Gibler, D. M. & Braithwaite, A., "Hostile Contiguous States" (Harvard Dataverse doi:10.7910/DVN/HQLOOJ) — a binary-response regression for whether a dyadic militarized dispute becomes **fatal**, on dyadic predictors. This is *interstate* conflict, distinct from the civil-war cases; fatal disputes are rare (~1.3%) → scored by **AUC-PR**.

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
- **Data file:** [`jays_clean.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/jays_clean.csv)
- **Notebook:** [`notebooks/06_pinyonjay_rsf_day2.ipynb`](notebooks/06_pinyonjay_rsf_day2.ipynb)
- **Published study:** Daw, S. K. et al. (2022). "Behavior-specific occurrence patterns of Pinyon Jays." *PLoS ONE* (doi:10.1371/journal.pone.0237621; S1 Dataset, open). A **logistic resource-selection function (RSF)** — habitat *use* points (1) vs *available* points (0) on environmental predictors. Use is the minority class, so out-of-sample skill is scored by **AUC-PR** (area under the precision-recall curve).

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
- **Data file:** [`snake_clean.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/snake_clean.csv)
- **Notebook:** [`notebooks/07_snake_rsf_day2.ipynb`](notebooks/07_snake_rsf_day2.ipynb)
- **Published study:** Multiscale spatial model of snake habitat selection, *PLoS ONE* (doi:10.1371/journal.pone.0123307; S1 Dataset, open). A **logistic resource-selection function** — *used* (1) vs *available* (0) locations (n = 2075) on canopy cover, shrub cover, aspect, elevation and ground cover. Use-vs-availability is the ecology workhorse design → scored by **AUC-PR**.

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

## 08 — Ames house prices  (Day 1)

- **Field:** Real estate / hedonic economics
- **Method:** OLS hedonic regression → lasso / ridge / elastic net (1-SE) + ABESS; held-out test
- **Data file:** [`AmesHousing.txt`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/AmesHousing.txt)
- **Notebook:** [`notebooks/08_ames_housing_day1.ipynb`](notebooks/08_ames_housing_day1.ipynb)
- **Published study:** De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education* 19(3). The analysis is an **OLS hedonic price regression** of sale price on property features (n = 2930; data at the JSE site). This example illustrates **regularization and variable selection** in a wide model.

**Unit of analysis:** a single-family residential **home sale**; n = 2,930. The printed hedonic regression
uses ten interpretable features; the regularization section uses De Cock's full ~80-feature set.

| Variable | Definition |
|---|---|
| `SalePrice` | sale price in USD **(outcome)** |
| `Overall.Qual` | overall material-and-finish quality (1-10) |
| `Gr.Liv.Area` | above-grade living area (sq ft) |
| `Total.Bsmt.SF` | total basement area (sq ft) |
| `Garage.Cars` | garage capacity (number of cars) |
| `Year.Built` | year of original construction |
| `Year.Remod.Add` | year of the last remodel/addition |
| `Full.Bath` | full bathrooms above grade |
| `TotRms.AbvGrd` | total rooms above grade (excludes bathrooms) |
| `Lot.Area` | lot size (sq ft) |
| `Fireplaces` | number of fireplaces |

---

## 09 — Building energy efficiency  (Day 1)

- **Field:** Engineering
- **Method:** Linear regression → polynomial regression (CV-selected degree); held-out test
- **Data file:** [`ENB2012_data.xlsx`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/ENB2012_data.xlsx)
- **Notebook:** [`notebooks/09_energy_efficiency_day1.ipynb`](notebooks/09_energy_efficiency_day1.ipynb)
- **Published study:** Tsanas, A. & Xifara, A. (2012). "Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools." *Energy and Buildings* 49:560–567. The paper's baseline is a **linear regression** predicting a building's heating load from 8 geometry inputs (n = 768; UCI id 242). This example illustrates the first regression extension: **polynomial terms**.

**Unit of analysis:** a simulated residential **building design** (12 shapes and variants, via Ecotect); n = 768.

| Variable | Definition |
|---|---|
| `Y1` | heating load (kWh per m^2 of floor area) **(outcome)** |
| `X1` | relative compactness (volume-to-surface ratio, normalized) |
| `X2` | surface area (m^2) |
| `X3` | wall area (m^2) |
| `X4` | roof area (m^2) |
| `X5` | overall height (m) |
| `X6` | orientation (2/3/4/5 = N/E/S/W) |
| `X7` | glazing area (fraction of floor area) |
| `X8` | glazing-area distribution (0-5 scheme) |

---

## 10 — Wine quality  (Day 1)

- **Field:** Food science / chemometrics
- **Method:** Multiple regression → polynomial regression (CV-selected degree); held-out test
- **Data file:** [`winequality-red.csv`](https://raw.githubusercontent.com/bdesmarais/intro-ml-istanbul_bilgi-2day/main/data/winequality-red.csv)
- **Notebook:** [`notebooks/10_wine_quality_day1.ipynb`](notebooks/10_wine_quality_day1.ipynb)
- **Published study:** Cortez, P. et al. (2009). "Modeling wine preferences by data mining from physicochemical properties." *Decision Support Systems* 47(4):547–553. The baseline is a **multiple (linear) regression** of sensory quality on 11 physicochemical measurements. Data: UCI (Vinho Verde red wine, n = 1599). This example illustrates **polynomial regression** — and using cross-validation to judge whether it is worth it.

**Unit of analysis:** an individual red 'Vinho Verde' **wine sample**; n = 1,599.

| Variable | Definition |
|---|---|
| `quality` | sensory quality, median of >=3 blind tasters (0-10) **(outcome)** |
| `fixed.acidity` | fixed (tartaric) acidity (g/L) |
| `volatile.acidity` | volatile (acetic) acidity (g/L) |
| `citric.acid` | citric acid (g/L) |
| `residual.sugar` | residual sugar (g/L) |
| `chlorides` | chlorides / salt (g/L) |
| `free.sulfur.dioxide` | free SO2 (mg/L) |
| `total.sulfur.dioxide` | total SO2 (mg/L) |
| `density` | density (g/mL) |
| `pH` | pH |
| `sulphates` | potassium sulphate additive (g/L) |
| `alcohol` | alcohol (% by volume) |
