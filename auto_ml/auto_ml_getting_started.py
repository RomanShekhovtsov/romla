from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

df_train, df_test = get_boston_dataset()
column_descriptions = {
    'MEDV': 'output',
    'CHAS': 'categorical'
}

ml_predictor = Predictor( type_of_estimator='regressor', column_descriptions= column_descriptions )
ml_predictor.train(df_train)

test_score = ml_predictor.score(df_test, df_test.MEDV)
file_name = ml_predictor.save()

trained_model = load_ml_model(file_name)
prediction = trained_model.predict(df_test)
print(prediction)
#ml_predictor.score(df_test, df_test.MEDV)
