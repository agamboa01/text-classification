import logging
import pandas as pd
from models.MachineLearningAlgorithm import MachineLearningAlgorithm

df = pd.read_csv("data/train_data.csv")
ml_model = MachineLearningAlgorithm()
logging.basicConfig(level = logging.INFO)


def test_MlAlgo():
    try:
        vectorizer, X_train, X_test, y_train, y_test = ml_model.vectorizer(df, text_column="TITLE",
                                                                           target_column="CATEGORY")
        logging.info('vectorized data is ready!')
        logging.info('Testing....')
        model, score = ml_model.LogisticRegressionModel(X_train, X_test, y_train, y_test)
        logging.info('model is ready!')
        ml_model.predict_target(vectorizer, text="Facebook to use drones, satellites to provide Internet everywhere",model=model)

    except:
        logging.error('error in vectorizing the data')


def test_DlAlgo():
    pass


if __name__ == "__main__":
    test_MlAlgo()