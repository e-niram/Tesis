/* ============================================================
Rolling Forecast - Daily Noise Level Prediction

Implements an expanding-window rolling forecast using
Exponential Smoothing (PROC ESM).

Parameters (edit the calls at the bottom of this file):
train_obs  : initial training window size in days (default 3500)
lead       : forecast horizon in days             (default 14)
target_var : column name in the input dataset
cluster_id : label used in output file names (e.g. cluster_0)
model_name : ESM model type — double | seasonal | winters | addwinters

Output datasets (work library):
work.all_forecasts    - every individual forecast with CI and actual
work.forecast_metrics - MAE, RMSE, MAPE per window

CSV exports (persistent across ODA sessions):
/home/u64274668/sasuser.v94/rolling_forecasts_daytime_<cluster_id>_<model_name>.csv
/home/u64274668/sasuser.v94/forecast_metrics_daytime_<cluster_id>_<model_name>.csv

Performance notes:
- Input data is passed directly to PROC ESM via OBS=, eliminating
the intermediate train_data copy (saves one I/O per window).
- PROC ESM is O(n) in the training window size; 100 windows on
3500+ observations completes in a few minutes on ODA.
============================================================ */
LIBNAME mydata "/home/u64274668/TFM/Datos/";

/* ---------------------------------------------------------- */
%MACRO rolling_forecast(input_ds, target_var, cluster_id=cluster_0,
   model_name=double, time_period=daytime, train_obs=3500, lead=14);

   %LOCAL total_obs current_end window_num;

   /* --- 1. Total observation count --- */
   PROC SQL NOPRINT;
      SELECT COUNT(*) INTO :total_obs TRIMMED FROM &input_ds.;
   QUIT;

   %PUT NOTE: Dataset=&input_ds. | Cluster=&cluster_id. | Model=&model_name. |
      Total obs=&total_obs.;
   %PUT NOTE: Train window=&train_obs. | Lead=&lead.;
   %PUT NOTE: Expected windows=%EVAL((&total_obs. - &train_obs.) / &lead.);

   /* --- 2. Initialize output datasets --- */
   PROC DATASETS LIB=WORK NOLIST;
      DELETE all_forecasts forecast_metrics esm_out window_forecasts
         window_actuals window_eval;
   QUIT;

   %LET current_end=&train_obs.;
   %LET window_num=1;

   /* Suppress per-iteration output to keep the log manageable */
   ODS GRAPHICS OFF;
   ODS RESULTS OFF;

   /* --- 3. Rolling forecast loop --- */
   /* Only iterate when a full LEAD window is available for evaluation */
   %DO %WHILE (%SYSEVALF(&current_end. + &lead. <= &total_obs.));

      %PUT NOTE: Window &window_num. | Training on obs 1-&current_end. |
         Forecasting obs %EVAL(&current_end.+1)-%EVAL(&current_end.+&lead.);

      /* 3a. Fit ESM on the training window using the requested model.
      OBS= passes only the training rows to PROC ESM directly,
      avoiding an intermediate data copy (saves one I/O per window).
      OUTFOR contains n_training in-sample rows followed by LEAD
      out-of-sample rows.
      Supported MODEL= values: double | seasonal | winters | addwinters */
      PROC ESM DATA=&input_ds.(OBS=&current_end.) OUTFOR=work.esm_out
         LEAD=&lead.;
         ID fecha_n INTERVAL=day;
         FORECAST "&target_var."n / MODEL=&model_name.;
      RUN;

      /* 3b. Extract the LEAD forecast rows from OUTFOR.
      OUTFOR has exactly (n_training + lead) rows, so
      FIRSTOBS=current_end+1 lands directly on the forecast rows.
      PREDICT / LOWER / UPPER are the OUTFOR column names for the
      forecast value and 95% confidence bounds.                    */
      DATA work.window_forecasts;
         SET work.esm_out(FIRSTOBS=%EVAL(&current_end. + 1));
         forecast_val=PREDICT;
         lower_ci=LOWER;
         upper_ci=UPPER;
         KEEP fecha_n forecast_val lower_ci upper_ci;
         FORMAT fecha_n DATE9.;
      RUN;

      /* 3c. Extract actual observed values for the forecast period */
      DATA work.window_actuals;
         SET &input_ds.(FIRSTOBS=%EVAL(&current_end. + 1)
            OBS=%EVAL(&current_end. + &lead.));
         actual="&target_var."n;
         KEEP fecha_n actual;
         FORMAT fecha_n DATE9.;
      RUN;

      /* 3d. Merge forecasts with actuals and compute error metrics */
      PROC SORT DATA=work.window_forecasts;
         BY fecha_n;
      RUN;

      PROC SORT DATA=work.window_actuals;
         BY fecha_n;
      RUN;

      DATA work.window_eval;
         MERGE work.window_forecasts (IN=inf) work.window_actuals (IN=ina);
         BY fecha_n;
         IF inf; /* left-join: retain all forecast rows */
         error=actual - forecast_val;
         abs_error=ABS(error);
         sq_error=error ** 2;
         IF actual > 0 THEN pct_error=ABS(error / actual) * 100;
         ELSE pct_error=.;

         window_num=&window_num.;
         train_end_obs=&current_end.;

         LABEL fecha_n="Date" forecast_val="Forecast (dB)" lower_ci=
            "95% Lower CI" upper_ci="95% Upper CI" actual="Actual (dB)" error=
            "Error (actual - forecast)" abs_error="Absolute Error (dB)" sq_error
            ="Squared Error" pct_error="Absolute % Error" window_num=
            "Window Number" train_end_obs="Last Training Observation";
      RUN;

      /* 3e. Append this window to the master results table */
      PROC APPEND BASE=work.all_forecasts DATA=work.window_eval FORCE;
      RUN;

      /* 3f. Advance the training window by LEAD observations */
      %LET current_end=%EVAL(&current_end. + &lead.);
      %LET window_num=%EVAL(&window_num. + 1);

   %END;

   /* Re-enable output */
   ODS RESULTS ON;
   ODS GRAPHICS ON;

   %PUT NOTE: Rolling forecast complete. %EVAL(&window_num. - 1) windows
      processed.;

   /* --- 4. Per-window summary metrics --- */
   PROC SQL;
      CREATE TABLE work.forecast_metrics AS SELECT window_num, train_end_obs,
         MIN(fecha_n) AS forecast_start FORMAT=DATE9. LABEL="Forecast Start",
         MAX(fecha_n) AS forecast_end FORMAT=DATE9. LABEL="Forecast End",
         COUNT(*) AS n LABEL="N Forecasts", MEAN(abs_error) AS MAE FORMAT=8.4
         LABEL="MAE (dB)", MEAN(sq_error) AS MSE FORMAT=8.4 LABEL="MSE (dB^2)",
         SQRT(MEAN(sq_error)) AS RMSE FORMAT=8.4 LABEL="RMSE (dB)",
         MEAN(pct_error) AS MAPE FORMAT=8.4 LABEL="MAPE (%)" FROM
         work.all_forecasts WHERE actual IS NOT MISSING GROUP BY window_num,
         train_end_obs ORDER BY window_num;
   QUIT;

   /* --- 5. Export to persistent CSV files (downloadable from ODA) --- */
   PROC EXPORT DATA=work.all_forecasts
      OUTFILE="/home/u64274668/sasuser.v94/rolling_forecasts_&time_period._&cluster_id._&model_name..csv"
      DBMS=CSV REPLACE;
   RUN;

   PROC EXPORT DATA=work.forecast_metrics
      OUTFILE="/home/u64274668/sasuser.v94/forecast_metrics_&time_period._&cluster_id._&model_name..csv"
      DBMS=CSV REPLACE;
   RUN;

   %PUT NOTE: Exported &cluster_id. / &model_name. to
      /home/u64274668/sasuser.v94/;

%MEND rolling_forecast;

/* ============================================================
Execute the rolling forecast for all clusters x all ESM models.
target_var : column name in TFM.ClusterMeansDaytime
cluster_id : used in output CSV file names
model_name : double | seasonal | winters | addwinters
train_obs  : 3500 = almost ten years of initial training data
lead       : 14  = predict the next 14 days each window
============================================================ */
/* --- cluster_0 (column "1"n) --- */
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_0, cluster_id=cluster_0,
   model_name=double, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_0, cluster_id=cluster_0,
   model_name=seasonal, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_0, cluster_id=cluster_0,
   model_name=winters, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_0, cluster_id=cluster_0,
   model_name=addwinters, train_obs=3500, lead=14);

/* --- cluster_1 (column "2"n) --- */
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_1, cluster_id=cluster_1,
   model_name=double, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_1, cluster_id=cluster_1,
   model_name=seasonal, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_1, cluster_id=cluster_1,
   model_name=winters, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_1, cluster_id=cluster_1,
   model_name=addwinters, train_obs=3500, lead=14);

/* --- cluster_2 (column "3"n) --- */
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_2, cluster_id=cluster_2,
   model_name=double, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_2, cluster_id=cluster_2,
   model_name=seasonal, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_2, cluster_id=cluster_2,
   model_name=winters, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansDaytime, Cluster_2, cluster_id=cluster_2,
   model_name=addwinters, train_obs=3500, lead=14);

/* --- nighttime / cluster_0 --- */
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_0, cluster_id=cluster_0,
   model_name=double,     time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_0, cluster_id=cluster_0,
   model_name=seasonal,   time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_0, cluster_id=cluster_0,
   model_name=winters,    time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_0, cluster_id=cluster_0,
   model_name=addwinters, time_period=nighttime, train_obs=3500, lead=14);

/* --- nighttime / cluster_1 --- */
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_1, cluster_id=cluster_1,
   model_name=double,     time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_1, cluster_id=cluster_1,
   model_name=seasonal,   time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_1, cluster_id=cluster_1,
   model_name=winters,    time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_1, cluster_id=cluster_1,
   model_name=addwinters, time_period=nighttime, train_obs=3500, lead=14);

/* --- nighttime / cluster_2 --- */
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_2, cluster_id=cluster_2,
   model_name=double,     time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_2, cluster_id=cluster_2,
   model_name=seasonal,   time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_2, cluster_id=cluster_2,
   model_name=winters,    time_period=nighttime, train_obs=3500, lead=14);
%rolling_forecast(TFM.ClusterMeansNighttime, Cluster_2, cluster_id=cluster_2,
   model_name=addwinters, time_period=nighttime, train_obs=3500, lead=14);

/* ============================================================
Display results (last cluster/model processed)
============================================================ */

/* First 2 windows (28 rows) — includes 95% confidence interval */
PROC PRINT DATA=work.all_forecasts(OBS=28) LABEL NOOBS;
   TITLE "Rolling Forecast Results - First 2 Windows (ESM)";
   VAR fecha_n window_num forecast_val lower_ci upper_ci actual error abs_error
      pct_error;
   FORMAT forecast_val lower_ci upper_ci actual error abs_error 8.2 pct_error
      8.2;
RUN;

/* Per-window accuracy metrics (first 10 windows) */
PROC PRINT DATA=work.forecast_metrics(OBS=10) LABEL NOOBS;
   TITLE "Forecast Accuracy Metrics - First 10 Windows (ESM)";
RUN;

/* Overall accuracy across all windows */
PROC MEANS DATA=work.all_forecasts N MEAN STDDEV MIN MAX;
   VAR abs_error pct_error;
   LABEL abs_error="Absolute Error (dB)" pct_error="Absolute % Error";
   TITLE "Overall Forecast Accuracy Summary - all windows (ESM)";
RUN;
