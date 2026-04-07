/* ============================================================
Rolling Forecast - Daily Noise Level Prediction

Implements an expanding-window rolling forecast using
ARIMA(1,1,1) (PROC ARIMA).

Parameters (edit the call at the bottom of this file):
   train_obs  : initial training window size in days (default 365)
   lead       : forecast horizon in days             (default 14)
   target_var : station column name                  (default 1)

Model specification:
   ARIMA(1,1,1) — one autoregressive term, one moving-average term,
   one regular difference.  First-differencing removes linear trend;
   AR(1) + MA(1) capture short-range autocorrelation.
   Estimated with Conditional Least Squares (METHOD=CLS) for speed.
   Adjust P=, Q= in the ESTIMATE statement for a different order.

Output datasets (work library):
   work.all_forecasts    - every individual forecast with CI and actual
   work.forecast_metrics - MAE, RMSE, MAPE per window

CSV exports (persistent across ODA sessions):
   /home/u64274668/sasuser.v94/rolling_forecasts_arima.csv
   /home/u64274668/sasuser.v94/forecast_metrics_arima.csv

Performance notes:
   - Input data is passed directly to PROC ARIMA via OBS=, eliminating
     an intermediate train_data copy (saves one I/O per window).
   - METHOD=CLS is O(n) and substantially faster than ML; MAXITER=20
     caps iteration cost per window.
   - 100 windows on 3500+ observations typically completes in under
     five minutes on ODA hardware.
============================================================ */
LIBNAME mydata "/home/u64274668/TFM/Datos/";

/* ---------------------------------------------------------- */
%MACRO rolling_forecast_arima(input_ds, target_var, train_obs=365, lead=14);

   %LOCAL total_obs current_end window_num;

   /* --- 1. Total observation count --- */
   PROC SQL NOPRINT;
      SELECT COUNT(*) INTO :total_obs TRIMMED FROM &input_ds.;
   QUIT;

   %PUT NOTE: Dataset=&input_ds. | Total obs=&total_obs.;
   %PUT NOTE: Train window=&train_obs. | Lead=&lead.;
   %PUT NOTE: Expected windows=%EVAL((&total_obs. - &train_obs.) / &lead.);

   /* --- 2. Initialize output datasets --- */
   PROC DATASETS LIB=WORK NOLIST;
      DELETE all_forecasts_arima forecast_metrics_arima arima_out
         window_forecasts window_actuals window_eval;
   QUIT;

   %LET current_end = &train_obs.;
   %LET window_num  = 1;

   /* Suppress per-iteration output to keep the log manageable */
   ODS GRAPHICS OFF;
   ODS RESULTS OFF;

   /* --- 3. Rolling forecast loop --- */
   /* Only iterate when a full LEAD window is available for evaluation */
   %DO %WHILE (%SYSEVALF(&current_end. + &lead. <= &total_obs.));

      %PUT NOTE: Window &window_num. | Training on obs 1-&current_end. |
         Forecasting obs %EVAL(&current_end.+1)-%EVAL(&current_end.+&lead.);

      /* 3a. Fit ARIMA(1,1,1) on the training window.
         OBS= passes only the training rows to PROC ARIMA directly,
         avoiding an intermediate data copy (saves one I/O per window).
         IDENTIFY applies one regular difference (d=1).
         ESTIMATE fits AR(1)+MA(1) via Conditional Least Squares.
         MAXITER is left at the default (50) to avoid convergence failures
         on windows where the series is harder to fit.
         FORECAST OUT= contains n_training in-sample rows followed by
         LEAD out-of-sample rows; PROC ARIMA back-transforms forecasts
         to the original (undifferenced) scale automatically.
         Output is suppressed via ODS RESULTS OFF set before the loop. */
      PROC ARIMA DATA=&input_ds.(OBS=&current_end.);
         IDENTIFY VAR="&target_var."n(1);
         ESTIMATE P=1 Q=1 METHOD=CLS;
         FORECAST LEAD=&lead. OUT=work.arima_out ID=fecha_n INTERVAL=day;
      QUIT;

      /* Guard: if estimation failed arima_out will have 0 rows.
         Skip steps 3b-3e for this window rather than erroring out. */
      %LOCAL arima_ok;
      %LET arima_ok = 0;
      PROC SQL NOPRINT;
         SELECT COUNT(*) INTO :arima_ok TRIMMED FROM work.arima_out;
      QUIT;

      %IF &arima_ok. = 0 %THEN %DO;
         %PUT WARNING: Window &window_num. - ARIMA estimation failed, window skipped.;
      %END;
      %ELSE %DO;

      /* 3b. Extract the LEAD forecast rows from the FORECAST OUT= dataset.
         OUT= has exactly (n_training + lead) rows, so
         FIRSTOBS=current_end+1 lands directly on the forecast rows.
         FORECAST / L95 / U95 are the OUT= column names for the
         forecast value and 95% confidence bounds.                     */
      DATA work.window_forecasts;
         SET work.arima_out(FIRSTOBS=%EVAL(&current_end. + 1));
         forecast_val = FORECAST;
         lower_ci     = L95;
         upper_ci     = U95;
         KEEP fecha_n forecast_val lower_ci upper_ci;
         FORMAT fecha_n DATE9.;
      RUN;

      /* 3c. Extract actual observed values for the forecast period */
      DATA work.window_actuals;
         SET &input_ds.(FIRSTOBS=%EVAL(&current_end. + 1)
            OBS=%EVAL(&current_end. + &lead.));
         actual = "&target_var."n;
         KEEP fecha_n actual;
         FORMAT fecha_n DATE9.;
      RUN;

      /* 3d. Merge forecasts with actuals and compute error metrics */
      PROC SORT DATA=work.window_forecasts; BY fecha_n; RUN;
      PROC SORT DATA=work.window_actuals;   BY fecha_n; RUN;

      DATA work.window_eval;
         MERGE work.window_forecasts (IN=inf)
               work.window_actuals   (IN=ina);
         BY fecha_n;
         IF inf;   /* left-join: retain all forecast rows */

         error     = actual - forecast_val;
         abs_error = ABS(error);
         sq_error  = error ** 2;
         IF actual > 0 THEN pct_error = ABS(error / actual) * 100;
         ELSE pct_error = .;

         window_num    = &window_num.;
         train_end_obs = &current_end.;

         LABEL fecha_n       = "Date"
               forecast_val  = "Forecast (dB)"
               lower_ci      = "95% Lower CI"
               upper_ci      = "95% Upper CI"
               actual        = "Actual (dB)"
               error         = "Error (actual - forecast)"
               abs_error     = "Absolute Error (dB)"
               sq_error      = "Squared Error"
               pct_error     = "Absolute % Error"
               window_num    = "Window Number"
               train_end_obs = "Last Training Observation";
      RUN;

      /* 3e. Append this window to the master results table */
      PROC APPEND BASE=work.all_forecasts_arima DATA=work.window_eval FORCE;
      RUN;

      %END; /* end %ELSE %DO (arima_ok > 0) */

      /* 3f. Advance the training window by LEAD observations */
      %LET current_end = %EVAL(&current_end. + &lead.);
      %LET window_num  = %EVAL(&window_num.  + 1);

   %END;

   /* Re-enable output */
   ODS RESULTS ON;
   ODS GRAPHICS ON;

   %PUT NOTE: Rolling forecast complete. %EVAL(&window_num. - 1) windows processed.;

   /* --- 4. Per-window summary metrics --- */
   PROC SQL;
      CREATE TABLE work.forecast_metrics_arima AS
         SELECT   window_num,
                  train_end_obs,
                  MIN(fecha_n)         AS forecast_start FORMAT=DATE9.
                                       LABEL="Forecast Start",
                  MAX(fecha_n)         AS forecast_end   FORMAT=DATE9.
                                       LABEL="Forecast End",
                  COUNT(*)             AS n              LABEL="N Forecasts",
                  MEAN(abs_error)      AS MAE            FORMAT=8.4
                                       LABEL="MAE (dB)",
                  SQRT(MEAN(sq_error)) AS RMSE           FORMAT=8.4
                                       LABEL="RMSE (dB)",
                  MEAN(pct_error)      AS MAPE           FORMAT=8.4
                                       LABEL="MAPE (%)"
         FROM     work.all_forecasts_arima
         WHERE    actual IS NOT MISSING
         GROUP BY window_num, train_end_obs
         ORDER BY window_num;
   QUIT;

   /* --- 5. Export to persistent CSV files (downloadable from ODA) --- */
   PROC EXPORT DATA=work.all_forecasts_arima
      OUTFILE="/home/u64274668/sasuser.v94/rolling_forecasts_arima.csv"
      DBMS=CSV REPLACE;
   RUN;

   PROC EXPORT DATA=work.forecast_metrics_arima
      OUTFILE="/home/u64274668/sasuser.v94/forecast_metrics_arima.csv"
      DBMS=CSV REPLACE;
   RUN;

   %PUT NOTE: Exported to /home/u64274668/sasuser.v94/;

%MEND rolling_forecast_arima;

/* ============================================================
Execute the rolling forecast
   target_var : station column (1 = Paseo de Recoletos)
   train_obs  : 365 = one year of initial training data
   lead       : 14  = predict the next 14 days each window
============================================================ */
%rolling_forecast_arima(TFM.SimpleForecasts, 1, train_obs=365, lead=14);

/* ============================================================
Display results
============================================================ */

/* First 2 windows (28 rows) — includes 95% confidence interval */
PROC PRINT DATA=work.all_forecasts_arima(OBS=28) LABEL NOOBS;
   TITLE "Rolling Forecast Results - First 2 Windows (ARIMA)";
   VAR fecha_n window_num forecast_val lower_ci upper_ci actual error
       abs_error pct_error;
   FORMAT forecast_val lower_ci upper_ci actual error abs_error 8.2
          pct_error 8.2;
RUN;

/* Per-window accuracy metrics (first 10 windows) */
PROC PRINT DATA=work.forecast_metrics_arima(OBS=10) LABEL NOOBS;
   TITLE "Forecast Accuracy Metrics - First 10 Windows (ARIMA)";
RUN;

/* Overall accuracy across all windows */
PROC MEANS DATA=work.all_forecasts_arima N MEAN STDDEV MIN MAX;
   VAR abs_error pct_error;
   LABEL abs_error = "Absolute Error (dB)"
         pct_error = "Absolute % Error";
   TITLE "Overall Forecast Accuracy Summary - all windows (ARIMA)";
RUN;
