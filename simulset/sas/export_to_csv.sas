/* Stage 0: export a SAS dataset to CSV for the Python profiling/generation
   pipeline. Run once in SAS 9.4; real.csv is then the sole input to
   profile_dataset.py. Keeping this step in SAS avoids a sas7bdat-reading
   dependency in Python and sidesteps SAS format/informat quirks.

   Note: this writes real.csv in your SAS session's encoding (check with
   %put &sysencoding;), not necessarily utf-8. Pass the matching
   --input-encoding to profile_dataset.py / generate_dataset.py on the
   Python side - see README's Stage 0 section. */

libname src "/path/to/sas/library";

proc export data=src.your_dataset
    outfile="/path/to/real.csv"
    dbms=csv
    replace;
run;
