# SQPP Streamlit dashboard

Read-only UI for the SQL Query Performance Predictor pipeline: explore timed queries, view model reports, and run predictions using `artifacts/best_model.joblib`.

## Prerequisites

Run the pipeline once from the **project root** (parent of this folder):

```bash
python setup_bird.py
python -u main.py
```

That produces `data/`, `reports/`, and `artifacts/` next to `streamlit_app/`.

## Install and run

From the project root:

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

## Point at another project directory

If you moved the repo or run Streamlit from elsewhere:

```powershell
$env:SQPP_PROJECT_ROOT = "c:\path\to\Advanced ML Course"
streamlit run app.py
```

The app loads:

- `data/query_dataset_raw.csv`, `data/query_dataset_features.csv`
- `reports/model_results.txt`, `reports/per_*.csv`
- `artifacts/best_model.joblib`
- `config.py` (for `FEATURE_COLS`, split settings)

## Pages

1. **Overview** — counts, pipeline steps, summary metrics, dataset charts  
2. **Data Explorer** — runtime plots, box plots, correlation, scatter  
3. **Model Results** — CV bars, per-class metrics, importances, tables  
4. **Predict** — manual feature form or paste SQL (uses `src/features/extract_features.py` on the project root)
