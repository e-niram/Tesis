/* ============================================================
Rolling Forecast - Daily Noise Level Prediction

Implements an expanding-window rolling forecast using
Double Exponential Smoothing (PROC ESM).

Parameters (edit the call at the bottom of this file):
train_obs : initial training window size in days (default 365)
lead      : forecast horizon in days            (default 14)
target_var: station ID column name              (default 1)

Output datasets (work library):
work.all_forecasts    - every individual forecast with its actual value
work.forecast_metrics - MAE, RMSE, MAPE per window

CSV exports (persistent across ODA sessions):
/home/u64274668/sasuser.v94/rolling_forecasts.csv
/home/u64274668/sasuser.v94/forecast_metrics.csv
============================================================ */
libname mydata "/home/u64274668/TFM/Datos/";

/* ---------------------------------------------------------- */
%macro rolling_forecast(input_ds, target_var, train_obs=365, lead=14);

   %local total_obs current_end window_num;

   /* --- 1. Total observation count --- */
   proc sql noprint;
      select count(*) into :total_obs trimmed from &input_ds.;
   quit;

   %put NOTE: Dataset=&input_ds. | Total obs=&total_obs.;
   %put NOTE: Train window=&train_obs. | Lead=&lead.;
   %put NOTE: Expected windows=%eval((&total_obs. - &train_obs.) / &lead.);

   /* --- 2. Initialize output datasets --- */
   proc datasets lib=work nolist;
      delete all_forecasts forecast_metrics train_data esm_out window_forecasts
         window_actuals window_eval;
   quit;

   %let current_end=&train_obs.;
   %let window_num=1;

   /* Suppress per-iteration output to keep the log manageable */
   ods graphics off;
   ods results off;

   /* --- 3. Rolling forecast loop --- */
   /* Only iterate when a full LEAD window is available for evaluation */
   %do %while (%sysevalf(&current_end. + &lead. <= &total_obs.));

      %put NOTE: Window &window_num. | Training on obs 1-&current_end. |
         Forecasting obs %eval(&current_end.+1)-%eval(&current_end.+&lead.);

      /* 3a. Extract training data (obs 1 through current_end) */
      data work.train_data;
         set &input_ds.(obs=&current_end.);
      run;

      /* 3b. Fit Double ESM and produce LEAD-step-ahead forecasts.
      OUTFOR contains both 'Actual' rows (training fit)
      and 'Forecast' rows (the lead-period predictions).         */
      proc esm data=work.train_data outfor=work.esm_out lead=&lead.;
         id fecha_n interval=day;
         forecast "&target_var."n / model=double;
      run;

      /* 3c. Keep only the FORECAST rows from OUTFOR.
      OUTFOR has exactly (n_training + lead) rows: the first n_training
      rows are the in-sample fit and the last LEAD rows are the
      out-of-sample forecasts.  PREDICT / LOWER / UPPER are the correct
      column names in the OUTFOR wide format (not the series variable).  */
      data work.window_forecasts;
         set work.esm_out(firstobs=%eval(&current_end. + 1));
         forecast_val=PREDICT;
         lower_ci=LOWER;
         upper_ci=UPPER;
         keep fecha_n forecast_val lower_ci upper_ci;
         format fecha_n date9.;
      run;

      /* 3d. Extract actual observed values for the same future period */
      data work.window_actuals;
         set &input_ds.(firstobs=%eval(&current_end. + 1) obs=
            %eval(&current_end. + &lead.));
         actual="&target_var."n;
         keep fecha_n actual;
         format fecha_n date9.;
      run;

      /* 3e. Merge forecasts with actuals and compute error metrics */
      proc sort data=work.window_forecasts;
         by fecha_n;
      run;

      proc sort data=work.window_actuals;
         by fecha_n;
      run;

      data work.window_eval;
         merge work.window_forecasts (in=inf) work.window_actuals (in=ina);
         by fecha_n;
         if inf; /* retain all forecast rows */
         error=actual - forecast_val;
         abs_error=abs(error);
         sq_error=error ** 2;
         if actual > 0 then pct_error=abs(error / actual) * 100;
         else pct_error=.;

         window_num=&window_num.;
         train_end_obs=&current_end.;

         label fecha_n="Date" forecast_val="Forecast (dB)" actual="Actual (dB)"
            lower_ci="95% Lower CI" upper_ci="95% Upper CI" error=
            "Error (actual - forecast)" abs_error="Absolute Error (dB)" sq_error
            ="Squared Error" pct_error="Absolute % Error" window_num=
            "Window Number" train_end_obs="Last Training Observation";
      run;

      /* 3f. Append this window to the master results table */
      proc append base=work.all_forecasts data=work.window_eval force;
      run;

      /* 3g. Advance the training window by LEAD observations */
      %let current_end=%eval(&current_end. + &lead.);
      %let window_num=%eval(&window_num. + 1);

   %end;

   /* Re-enable output */
   ods results on;
   ods graphics on;

   %put NOTE: Rolling forecast complete. %eval(&window_num. - 1) windows
      processed.;

   /* --- 4. Per-window summary metrics --- */
   proc sql;
      create table work.forecast_metrics as select window_num, train_end_obs,
         min(fecha_n) as forecast_start format=date9. label="Forecast Start",
         max(fecha_n) as forecast_end format=date9. label="Forecast End",
         count(*) as n label="N Forecasts", mean(abs_error) as MAE format=8.4
         label="MAE (dB)", sqrt(mean(sq_error)) as RMSE format=8.4
         label="RMSE (dB)", mean(pct_error) as MAPE format=8.4 label="MAPE (%)"
         from work.all_forecasts where actual is not missing group by
         window_num, train_end_obs order by window_num;
   quit;

   /* --- 5. Export to persistent CSV files (downloadable from ODA) --- */
   proc export data=work.all_forecasts
      outfile="/home/u64274668/sasuser.v94/rolling_forecasts.csv" dbms=csv
      replace;
   run;

   proc export data=work.forecast_metrics
      outfile="/home/u64274668/sasuser.v94/forecast_metrics.csv" dbms=csv
      replace;
   run;

   %put NOTE: Exported to /home/u64274668/sasuser.v94/;

%mend rolling_forecast;

/* ============================================================
Execute the rolling forecast
- target_var : station column (1 = Paseo de Recoletos)
- train_obs  : 365 = one year of initial training data
- lead       : 14  = predict the next 14 days each window
============================================================ */
%rolling_forecast(TFM.SimpleForecasts, 1, train_obs=365, lead=14);

/* ============================================================
Display results
============================================================ */

/* First 2 windows (28 rows) */
proc print data=work.all_forecasts(obs=28) label noobs;
   title "Rolling Forecast Results - First 2 Windows";
   var fecha_n window_num forecast_val actual error abs_error pct_error;
   format forecast_val actual error abs_error 8.2 pct_error 8.2;
run;

/* Per-window accuracy metrics (first 10 windows) */
proc print data=work.forecast_metrics(obs=10) label noobs;
   title "Forecast Accuracy Metrics - First 10 Windows";
run;

/* Overall accuracy across all windows */
proc means data=work.all_forecasts n mean stddev min max;
   var abs_error pct_error;
   label abs_error="Absolute Error (dB)" pct_error="Absolute % Error";
   title "Overall Forecast Accuracy Summary (all windows)";
run;
