python train.py --mode %1 --train-csv %2\train.csv --model-dir . %3 %4
@IF NOT %ERRORLEVEL%==0 GOTO :EXIT
python predict.py --test-csv %2\test.csv --prediction-csv %2\predict.csv --model-dir . %3 %4
:EXIT
@EXIT /B %ERRORLEVEL%
