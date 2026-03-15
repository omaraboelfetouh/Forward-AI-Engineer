import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_mlops_pipeline():
    print("Starting MLOps Pipeline...")
    # Generate dummy data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = X.sum(axis=1) + np.random.rand(100) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        print(f"Model trained with MSE: {mse:.4f}")
        print("Model saved to MLflow registry.")

if __name__ == "__main__":
    run_mlops_pipeline()