# models.py
# contains all model related information used by training


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import Input
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
logger = logging.getLogger(__name__)

def build_lstm(input_shape, num_classes=3):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_lstm_model():
    return build_lstm((30, 1))

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs")
)

models = [
    (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs")),
        "Logistic Regression",
        "Simple linear baseline"
    ),
    (
        RandomForestClassifier(n_estimators=50, random_state=42),
        "Random Forest",
        "Splits data with many trees"
    ),
    (
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting",
        "Builds strong ensemble step by step"
    ),
    (
        make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        "SVM (RBF Kernel)",
        "Finds complex decision boundaries"
    ),
    (
        KerasClassifier(model=build_lstm_model, epochs=5, batch_size=32, verbose=0),
        "LSTM",
        "Learns patterns over time in sequences"
    )
]