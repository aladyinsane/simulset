/* Stage 0: export a SAS dataset to CSV for the Python profiling/generation
   pipeline. Run once in SAS 9.4; real.csv is then the sole input to
   profile_dataset.py. Keeping this step in SAS avoids a sas7bdat-reading
   dependency in Python and sidesteps SAS format/informat quirks. */

libname src "/path/to/sas/library";

proc export data=src.your_dataset
    outfile="/path/to/real.csv"
    dbms=csv
    replace;
run;
