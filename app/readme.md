# What was built

## Backend pipeline (3 new scripts)

**[src/pipeline/06_train_and_save_models.py](../src/pipeline/06_train_and_save_models.py)**
Run once (then monthly) to train a Random Forest per cluster per period (6 models total) on the full 4,018-observation dataset and save them to `models/` with joblib. Same feature engineering as `ml_predictions.py` (19 lags + 6 cyclic calendar features).

**[src/pipeline/04_fetch_api.py](../src/pipeline/04_fetch_api.py)**
Daily fetch from the CKAN API. Handles all format differences vs. `data/final/`:
- Merges `Año/mes/dia` → `FECHA`; converts comma decimals
- Computes `LAeqDiurno` = energy-mean of tipos D+E; `LAeqNocturno` = tipo N
- Imputes missing stations using seasonal neighbors (reuses `03_handle_missing.py` logic in pressure domain)
- Appends to `data/final/daytime_final.csv` and `nighttime_final.csv`

**[src/pipeline/05_update_predictions.py](../src/pipeline/05_update_predictions.py)**
Loads the saved `.pkl` models, updates cluster means with the new day, runs inference (no retraining), maps cluster forecasts to individual stations, and writes `app/data/predictions.json`.

## Frontend (5 files)

| File | Purpose |
|------|---------|
| [app/index.html](index.html) | Semantic HTML, WCAG 2.1 AA (skip links, ARIA labels, keyboard nav) |
| [app/css/style.css](css/style.css) | Madrid brand (`#003082` blue, Source Sans Pro), WHO colour thresholds |
| [app/js/map.js](js/map.js) | Leaflet.js with coloured circle markers, popup details |
| [app/js/charts.js](js/charts.js) | Chart.js 14-day forecast line chart with WHO reference lines |
| [app/js/app.js](js/app.js) | Controller — loads JSON, wires period/day controls, ranking table |

## Automation

- [.github/workflows/daily_update.yml](../.github/workflows/daily_update.yml) — Runs at 08:00 UTC, fetches yesterday's API data, updates predictions, commits
- [.github/workflows/monthly_retrain.yml](../.github/workflows/monthly_retrain.yml) — Re-trains models on the 1st of each month

## First-time setup checklist

1. Run `python src/pipeline/06_train_and_save_models.py` locally to generate the `.pkl` files
2. Commit the model files (Git LFS is configured via `.gitattributes`)
3. Run `python src/pipeline/05_update_predictions.py` to generate the initial `predictions.json`
4. Enable **GitHub Pages** in repo Settings → Pages → Source: `main` branch, `/app` folder
5. Enable **Git LFS** (`git lfs install`) if models exceed 50 MB

**Total cost: €0/month.**
