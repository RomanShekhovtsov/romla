python train.py --mode %1 --train-csv %2\train.csv --model-dir .
if ERRORLEVEL 0 python predict.py --test-csv %2\test.csv --prediction-csv %2\predict.csv --model-dir .