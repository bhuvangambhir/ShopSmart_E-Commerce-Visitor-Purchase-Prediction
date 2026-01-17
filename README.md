# E-commerce Revenue Prediction using Decision Tree Classifier

## Project Overview
This project aims to predict whether a user will generate revenue in an e-commerce session based on various behavioral and technical attributes. We utilize a Decision Tree Classifier, including hyperparameter tuning and cost-complexity pruning, to build an accurate predictive model.

## Dataset
The dataset used for this project is `shop_smart_ecommerce.csv`. It contains various features related to user sessions, such as page views, bounce rates, exit rates, administrative pages visited, and other session-specific metrics. The target variable is `Revenue`, indicating whether a purchase was made during the session.

### Features:
*   `Administrative`, `Administrative_Duration`: Number and duration of administrative pages visited.
*   `Informational`, `Informational_Duration`: Number and duration of informational pages visited.
*   `ProductRelated`, `ProductRelated_Duration`: Number and duration of product-related pages visited.
*   `BounceRates`, `ExitRates`, `PageValues`: Metrics related to user navigation and engagement.
*   `SpecialDay`: Closeness to a special day (e.g., Mother's Day, Valentine's Day).
*   `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`: Session context features.
*   `VisitorType`: Type of visitor (e.g., Returning_Visitor, New_Visitor).
*   `Weekend`: Boolean indicating if the session occurred on a weekend.
*   `Revenue`: Target variable (True if revenue was generated, False otherwise).

## Methodology
1.  **Data Loading & Initial Exploration**: Loaded the `shop_smart_ecommerce.csv` dataset into a pandas DataFrame. Performed initial checks for null values and data types.
2.  **Feature Encoding**: Categorical features (`Month`, `VisitorType`, `Weekend`, `Revenue`) were converted into numerical representations using `LabelEncoder`.
3.  **Data Splitting**: The dataset was split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) with a `test_size` of 0.2 and `random_state=42`.
4.  **Decision Tree Training & Hyperparameter Tuning**: A `DecisionTreeClassifier` was trained. Hyperparameter tuning was performed using a grid search approach to find the optimal `max_depth` and `min_samples_split` values that maximize accuracy on the test set. The optimal parameters found were `max_depth=8` and `min_samples_split=25`.
5.  **Cost-Complexity Pruning (CCP)**: The model was further optimized using cost-complexity pruning to avoid overfitting. The `ccp_alphas` were computed, and the best `alpha` value was selected based on the highest accuracy on the test set.
6.  **Model Evaluation**: The performance of the best models (tuned and pruned) was evaluated using:
    *   Accuracy Score
    *   Confusion Matrix
    *   Classification Report (Precision, Recall, F1-score)
    *   F1-score
7.  **Model Visualization**: The pruned decision tree was visualized to provide insight into its decision-making process.
8.  **Model Export**: The best performing Decision Tree model (`best_model_dt.pkl`) and the LabelEncoder (`le.pkl`) were saved using `joblib` for future use.

## Results

### Hyperparameter Tuned Model Results (max_depth=8, min_samples_split=25):
*   **Accuracy:** 0.8950
*   **F1-Score:** 0.6201

### Cost-Complexity Pruned Model Results (best_alpha=0.000298):
*   **Accuracy:** 0.8970
*   **F1-Score:** 0.6492

### Combined Tuned and Pruned Model Results (max_depth=8, min_samples_split=25, ccp_alpha=0.000298):
*   **Accuracy:** 0.8954
*   **F1-Score:** 0.6346

The cost-complexity pruned model (`best_model_dt`) showed slightly better performance in terms of accuracy and F1-score.

## How to Run
1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```
3.  **Download the Dataset**: Ensure `shop_smart_ecommerce.csv` is in the root directory or update the path in the notebook.
4.  **Open and Run the Jupyter Notebook**:
    Open `your_notebook_name.ipynb` in a Jupyter environment (e.g., Jupyter Lab, Google Colab) and run all cells.

## Files in this Repository
*   `your_notebook_name.ipynb`: The main Jupyter Notebook containing the analysis and model development.
*   `shop_smart_ecommerce.csv`: The dataset used in the project.
*   `best_model_dt.pkl`: The saved best performing Decision Tree Classifier model.
*   `le.pkl`: The saved LabelEncoder object used for data preprocessing.
*   `README.md`: This file.

