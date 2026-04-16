#  COM 763: Advanced Machine Learning (Task 1)

# SQL Query Runtime Prediction

**Author:** Aled Caio Rowlands  
**Module:** COM 763 Advanced Machine Learning

---

### Abstract

This report documents a machine learning pipeline for SQL query runtime prediction using the BIRD Mini-Dev text-to-SQL benchmark. The project began as binary classification (fast vs slow) and shifted to regression on `log(runtime_seconds)` after the classifier proved too weak.

The core finding was a clear limitation: SQL features alone do not transfer across unseen databases. Cross-schema classification stayed near-random and cross-schema regression produced negative `R2`. When schema statistics were added and evaluation was restricted to individual databases, performance improved sharply for well-indexed schemas i.e. **R²(log) ≈ 0.945** on `debit_card_specializing` and **≈ 0.929** on `formula_1` with Ridge (α = 10), whilst still failing on poorly indexed schemas like `financial`.

A fundamental error was not diagnosed until late into the project. I failed to EDA some key attibutes of my data (i.e size and distbution) until the architecture around the data was already built. By the time the class distributions and per-database imbalances became obvious, I had essentially caused a sunk cost fallacy, the pipeline, feature extraction, model scaffolding, streamlit app and deployment code were all wired to BIRD Mini-Dev. Switching datasets would have meant many hours of re-engineering with no guarantee of finding something similar, and increasing the data set was out of the scope of the project. 

The project had **three main iterations** of evaluation and modelling, with the **first iteration** split into **two uses**: **(1)** classifiers trained on seen databases and tested on held-out databases; **(1, subset use) {i don't undestand what supset use means i understand it's the same model tested on the seen databases but could be wrote better}** the same classifier and SQL feature pipeline, but a **subset** of that use case, a pooled global model after a per-database 80/20 query split so every schema contributes unseen queries to the test pool; **(2)** per-database SQL-only classification (often ill-posed given label skew); **(3)** per-database regression with schema statistics. All quantitative results cited below are collated in `reports/report_evidence_bundle.md` and `reports/report_metrics_long.csv` for traceability.

## 1. Problem Definition and System Framing

SQL query performance is difficult to judge from query text alone, yet slow queries have real costs: compute time, delayed dashboards, increased cloud spend, and manual profiling. A lightweight ML model that could estimate runtime before execution would have practical value.

I focused on a constrained version: can runtime be predicted from the structure of the SQL itself, without deep database integration? 

The central research question was:

> **Can a model trained on queries from some databases predict runtime on databases it has never seen?**

That question matters because usefulness depends on cross-schema transfer. If a model only works on schemas it has seen, every new deployment needs retraining and the tool loses its value.

The project did not keep a single formulation. I originally framed it as binary classification because it looked simple and deployable. Later evidence showed this was masking key problems, so the final formulation became regression on `log(runtime_seconds)`. Several limitations were not recognised early: all timings were collected on my laptop using SQLite, and the SQL-only feature set excludes row counts, indexes, and cache state — factors that later proved to be the main reason transfer failed.  

{this section could do with majjor improvements. Firstly the porject changed a lot the scope was too large and the idea of SQL quires without schme acould transfer is stupid it was acutalyl about conetxt and bird SQL had lot sof context i.e. data base ,stuff u know just search up bird and hgihlgit the ino that ould acutally have made a unseen db and teh queires transfer from noe seee, i..e only having the size of the data base but no quieres trained on it}.  
{finally disscuss moreabout the final product and what the streamlit app will do , i..e the streamlit app provides the final model on just a couple databases and alwos users to input wuries rnsit on aSQL lite db and also reuns the model on it. and allows you to see the stats of all models and test there runtimes on specific quires, and see how accurate tehy are, it runs a SQL lite server and yeah runs the model too on the query.}  

## 2. Data Pipeline and Feature Handling

The data source was **BIRD Mini-Dev**, a text-to-SQL benchmark covering 11 SQLite databases across finance, sport, education, etc. Although using a text-to-SQL benchmark for runtime prediction seems unusual, it provided realistic SQL across well-designed and labelled databases, and was small enough to run repeatedly on local hardware.

Each query was executed against its SQLite database with **3 timing runs** and a **30-second timeout**, taking the **median runtime** as the label.


| Snapshot                    | Raw rows | Labelled rows | Fast | Slow |
| --------------------------- | -------- | ------------- | ---- | ---- |
| Earlier baseline extraction | 425      | 320           | 213  | 107  |
| Current expanded snapshot   | 498      | 374           | 249  | 125  |


The (`data/query_dataset_features.csv`) is what the metrics in Section 3 use.

### The quantile bucketing decision

A key methodological decision was **quantile-based bucketing**: the bottom 50% of runtimes were labelled `fast`, the top 25% labelled `slow`, and the **middle 25% dropped entirely**. The reasoning: the middle bracket contains ambiguous queries that are neither clearly fast nor clearly slow. Including them would force the classifier to draw a line through a noisy, overlapping region where labels would be essentially arbitrary. Cutting the middle gave the model cleaner class boundaries.

The 50/25 split rather than 50/50 was deliberate. Slow queries are the ones that actually matter — they are the expensive, problematic ones users want warnings about. Setting a higher bar for "slow" (top 25%) meant the label captured genuinely expensive queries rather than anything above the median. The fast label at 50% was kept wider because there is less ambiguity at the bottom end.

That said, this removed **124 queries (24.9%)** of the raw rows at the current snapshot size. In hindsight, dropping those borderline examples was a structural weakness — they are exactly the cases a model needs to learn the fast-to-slow transition.

Figure 12 -- Query counts per database  
*Figure 12. Query counts per database, showing highly uneven across schemas.*


| Database                | Queries | Fast | Slow | % Slow |
| ----------------------- | ------- | ---- | ---- | ------ |
| superhero               | 50      | 50   | 0    | 0%     |
| card_games              | 50      | 7    | 43   | 86%    |
| student_club            | 47      | 47   | 0    | 0%     |
| european_football_2     | 45      | 8    | 37   | 82%    |
| formula_1               | 41      | 38   | 3    | 7%     |
| codebase_community      | 35      | 3    | 32   | 91%    |
| financial               | 32      | 28   | 4    | 12%    |
| thrombosis_prediction   | 27      | 27   | 0    | 0%     |
| toxicology              | 25      | 24   | 1    | 4%     |
| debit_card_specializing | 19      | 14   | 5    | 26%    |
| california_schools      | 3       | 3    | 0    | 0%     |


### The EDA mistake

To be blunt about this: I was stupid and forgot to do proper EDA until the entire architecture around the data was already built. The pipeline, feature extraction, training loop, evaluation harness, and Streamlit deployment were all wired to BIRD Mini-Dev before I ever looked at the per-database class distributions above. By the time I realised how broken the class balance was, with some databases with 0% slow or 90%+ slow. There was no way to switch to a different dataset without many hours of re-engineering, and no guarantee of finding data of a similar nature. Furthermore data quieres are poorly distbuted, and I could not in the scope of the project increase.   

{this needs to disscuss regression being used and less bullshsit about  the fucking buckets it's a issue but showuldn't take half the report. jesus, comon. This is the biggest issue need more dissucssion on how what featuers were used  

and also how in the end we added a feature of data base size and those features, epxlain the use case of that, and explain how we had already implelmented models, explain why calfornai schools is so small. Please add words but don't drag it out}

Figure 4 -- The dropped middle bracket
*Figure 4. The dropped middle 25% of runtimes.*

## 3. Model Implementation and Debugging

This section follows **three iterations**, with **iteration 1** documented in **two parts**: the **full** cross-schema evaluation, then a **subset of that same use** (same models and SQL features, different train/test geometry). The two parts of iteration 1 share one classifier helper in the repository: a **held-out database** split (`unseen`_* metrics) vs a **per-database 80/20 query split with pooled training** (`seen`_* metrics). The code path is `_evaluate_classifier_df` in `rerun_commit_comparison.py`, which calls `_split_seen` for the subset-use part.

### Iteration 1.2 — Held-out databases (cross-schema classification)

**Question:** Can a classifier trained only on **seen** databases predict fast vs slow on **held-out** databases, using the SQL structural feature vector only (no schema statistics)?

**Split:** Train on all databases except `financial` and `formula_1`; test only on those two. This matches `database_aware` splitting in `src/models/train.py` and the summary in `reports/split_summary.csv` (301 train queries, 73 test).

**Metrics:** The end-to-end pipeline (`python main.py`) selects **XGBoost** by cross-validation and evaluates on the holdout. On the current snapshot, **test F1 (slow) = 0.186**, **ROC-AUC = 0.461**, **accuracy = 0.52** (`reports/model_results.txt`). Per-database test breakdown: **financial F1 = 0.18**, **formula_1 F1 = 0.19**. Alternative families on the same split (see `reports/all_models_test_comparison.csv`) score worse on F1 for this task except logistic regression, which achieves higher F1 (0.32) but is not the CV-selected deployable choice in the main pipeline.

**Decision:** Transfer is poor; the original deployment question (“warn on a brand-new schema”) is not answered. The next step is **iteration 1 — subset use**: keep the **same** classifier setup and SQL features, but change **only** how train and test queries are formed so that every database contributes **unseen queries** while still training a **single global** model.  

{Shorten this itteration explain how especailly with the issue smenitoend above this was bound ot fail keep a couple stats, but not gun ho with it}

### Itertion 1.2, Unseen queries within each database (pooled global classifier)

**Question:** If each `db_id` is split **80/20** into train and test queries (stratified where both classes have enough support), and the per-database training portions are **pooled** into one training set — and similarly for the test portions — how strong is slow-class performance? This is a fairer check of “does the signal live in query text when schemas are not literally held out,” while still testing **unseen queries**.

**Method:** Exactly `**_split_seen`** in `rerun_commit_comparison.py`: per-database query-level split, then concatenate. One global classifier is fit on the pooled train rows and scored on the pooled test rows (`seen`_* outputs of `_evaluate_classifier_df`).

**Metrics (current `query_dataset_features.csv`):** pooled **296 train / 78 test** query rows; **seen** split — best model **Random Forest** (by internal CV in `_evaluate_split`), **F1 (slow) = 0.391**, **ROC-AUC = 0.64**, **accuracy = 0.641**. For reference, the **unseen-database** branch on the same file still yields **F1 = 0.174**, **ROC-AUC = 0.457** with **XGBoost** — confirming that most of the apparent “gain” comes from not holding out whole databases, not from a magic improvement in the feature space.

**Decision:** The SQL-only vector carries **some** ranking signal when test queries come from the same schemas as training, but per-database label skew (Section 2) still makes “fast vs slow” a confounded target, and iteration 1 (full holdout) already showed **cross-schema deployment** is not viable. The next **iteration** moves beyond this shared classifier setup.

Figure 1 -- Seen vs unseen performance
*Figure 1. Historical comparison across pipeline commits (supplementary). The **primary** iteration 1 (full) vs iteration 1 (subset use) numbers for the current labelled snapshot are the tables in this section and `reports/report_evidence_bundle.md`.*  

*{Cut this section largely explain how even on seen database models stats were terrible, and yeah}*

### Iteration 2 — Per-database SQL-only classification

**Question:** Can we obtain **usable classification** by fitting **separately per database** with SQL-only features (e.g. logistic regression per `db_id`)?

**Evidence in this repository:** There is **no** dedicated CSV that lists per-database logistic (or other) classifiers with an 80/20 split. What we **can** say from the data is that several databases have **0% slow** queries under the global quantile policy (Section 2 table), so the slow class is **undefined** for those schemas — any slow-F1 metric would be meaningless. Others are extremely skewed (e.g. 86–91% slow), so a global “fast vs slow” boundary mixes **schema difficulty** with **query shape**. In practice this iteration motivated **adding schema context** and moving to a **continuous** target for many diagnostics.

**Decision:** Treat per-database **classification** as **partially ill-posed** on this benchmark under global quantile labels; proceed to **iteration 3** (per-database **regression** with schema statistics).

### Iteration 3 — Per-database regression with schema statistics

**Question:** If we train **independently per database**, add **schema statistics** (table sizes, index coverage, etc.) alongside SQL features, and predict `**log(runtime)`**, do we recover strong metrics on some schemas?

**Method:** `run_schema_stats_model.py` — 80/20 split **within** each database; model zoo includes Ridge, Lasso, random forest, and gradient boosting. Full grid is in `reports/within_db_schema_metrics.csv`.

**Result (summary):** Strong **R²(log)** on e.g. `debit_card_specializing` and `formula_1`; weak or negative on others such as `financial` and several skewed schemas. See Section 4 for the full best-per-database table and figures for this iteration.

**Optional bridge:** **Cross-schema regression** on SQL-only features with the same database holdout still gives **R²(log) = −1.0469** for the best Ridge (α = 10) setting — worse than predicting the mean — so moving to a continuous target does **not** fix universal transfer. {yeah this is something to note that the real factor for improvemtn is the schemme feature input, explain that feature input was a result. explain what the features were agian this is something fundamentla any reader would be reading and say,,,,, 'well what did you use to tirain your model?' this should be epxlain in it 1 for the shcmean queir features but here should be explain the shcmean features i.e. data base stats.}

### {This is the model this is the actual product artifact this is the one to spend the msot nto least time on, smonn }

## 4. Experimental Evaluation and Model Selection

### Cross-schema verdict

Across both classification and regression: **the SQL-only model did not transfer to unseen schemas** in a way that would support a single global deployment.  


Figure 10 -- R2 comparison
*Figure 10. Unseen-schema regression remains below R² = 0 for SQL-only features.*

### Within-database schema-aware results (iteration 3 — detail)

Once schema statistics were introduced and each database was trained independently, a more nuanced picture emerged.

Figure 13 -- Best within-database R2
*Figure 13. Within-database performance varies sharply by schema.*


| Database                | n   | Slow % | Index coverage | Best model  | R² (log)    | MAE (s)  |
| ----------------------- | --- | ------ | -------------- | ----------- | ----------- | -------- |
| debit_card_specializing | 19  | 26%    | 0.67           | Ridge(a=10) | **0.945**   | 0.031    |
| formula_1               | 41  | 7%     | 0.50           | Ridge(a=10) | **0.929**   | 0.152    |
| student_club            | 47  | 0%     | 0.88           | Ridge(a=10) | **0.640**   | 0.000049 |
| european_football_2     | 45  | 82%    | 0.62           | Lasso       | **0.148**   | 0.120    |
| superhero               | 50  | 0%     | 0.00           | Lasso       | 0.099       | 0.000106 |
| toxicology              | 25  | 4%     | 1.00           | RF          | 0.009       | 0.001    |
| card_games              | 50  | 86%    | 0.29           | RF          | -0.329      | 0.195    |
| codebase_community      | 35  | 91%    | 0.38           | RF          | -0.390      | 0.230    |
| financial               | 32  | 12%    | 0.00           | RF          | **-10.435** | 0.007    |




{just chekc that htis is not randome varibliitly in R2 log perofmance, please just doible scehk}

Values are **best R²(log)** per database from `reports/within_db_schema_metrics.csv` (rounded for display where noted in the abstract).

Figure 14 -- Schema complexity versus R2
*Figure 14. Index coverage predicts model success more clearly than raw database size.*

This was the strongest evaluation evidence in the project. The feature set was not universally useless — its effectiveness depended on schema properties, particularly index coverage. Well-indexed databases with moderate query variety became highly predictable. Poorly indexed or highly skewed databases resisted prediction regardless of model choice.



{What is a well indenxed database , what feature is it ? EXPLAIN} 

## 5. Deployment

A Streamlit deployment was maintained as part of the end-to-end pipeline. The app lets a user paste a SQL query, trigger feature extraction, and receive a predicted runtime estimate.

The deployment should be interpreted as proof of system integration, not evidence that predictions are universally reliable. The best results depended on schema properties not available in a pure SQL-text-only deployment.

- **Streamlit deployment URL:** *Publish the app on [Streamlit Community Cloud](https://streamlit.io/cloud) and insert your public app URL in the submitted version of this report.*
- **GitHub repository URL:** [https://github.com/AledCaioRow/COM763---ML-pipeline-for-SQL-error-detection-project](https://github.com/AledCaioRow/COM763---ML-pipeline-for-SQL-error-detection-project)



## References

Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., Wang, B., Qin, B., Geng, R., Huo, N., Zhou, X., Ma, C., Huang, R., Lou, Q., Chen, Z., Zhang, Z., Li, Z., Zhu, J., Cai, T., Chen, R., Chen, X., Huang, S., Liu, K. and Zhu, Y. (2024). *Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQLs.* Advances in Neural Information Processing Systems (NeurIPS), 36. Available at: [https://arxiv.org/abs/2305.03111](https://arxiv.org/abs/2305.03111)

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp.2825-2830.

Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), pp.5-32.

Friedman, J.H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics, 29(5), pp.1189-1232.

Hoerl, A.E. and Kennard, R.W. (1970). *Ridge regression: biased estimation for nonorthogonal problems.* Technometrics, 12(1), pp.55-67.

Marcus, R., Negi, P., Mao, H., Zhang, C., Alizadeh, M., Kraska, T., Papaemmanouil, O. and Tatbul, N. (2019). *Neo: A Learned Query Optimizer.* Proceedings of the VLDB Endowment, 12(11), pp.1705-1718.

Streamlit Inc. (2024). *Streamlit Documentation.* Available at: [https://docs.streamlit.io](https://docs.streamlit.io)