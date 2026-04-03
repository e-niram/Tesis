libname mydata "/home/u64274668/DatosEjerciciosClase/Series/";

%macro rolling_forecast(input_ds, target_var, start_obs, lead_days);

   %global total_obs;

   /* 1. Get count via SQL */
   proc sql noprint;
      select count(*) into :total_obs from &input_ds.;
   quit;

   /* 2. Reset results */
   proc datasets lib=work nolist;
      delete final_forecasts;
   quit;

   %let current_end=&start_obs.;

   /* 3. The Loop */
   %do %while (%sysevalf(&current_end. < &total_obs.));

      data current_train;
         set &input_ds.(obs=&current_end.);
      run;

      /* Standardized PROC ESM for ODA */

      /* Note: Removed 'noprint' to avoid the syntax conflict with 'lead' */
      proc esm data=current_train outfor=temp_out lead=&lead_days.;
         id FECHA_n interval=day;
         forecast "&target_var."n / model=double;
      run;

      /* 4. Filter for Forecast using two methods (Name and Type) */
      data temp_forecast_only;
         set temp_out;
         /* Check for 'FORECAST' in the Status variable (_ST_ or _TYPE_) */
         if upcase(_ST_)='FORECAST' or "&target_var."n=.;

         /* Metadata */
         window_origin=&current_end.;
      run;

      /* Append to the master table */
      proc append base=work.final_forecasts data=temp_forecast_only force;
      run;

      /* Increment */
      %let current_end=%eval(&current_end. + &lead_days.);

   %end;

%mend rolling_forecast;

/* 4. Execute */
/* Ensure you use the column name '1' as the target variable */
%rolling_forecast(mydata.diurno, 1, 3500, 14);

/* 5. Results */
proc print data=work.final_forecasts(obs=30);
   title "Final Rolling Forecast Results";
run;
