# ML

## Table of Content

1. **Basics of ML**
    * Day 1 - [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML0.ipynb)
        * Train Test Split
        * Prediction using SVM
        * Support Vector Machine
        * Best model and hyper parameter tunning using GridSearchCV
        * K Fold Cross Validation
        * sklearn.model_selection.StratifiedKFold
    *  Day 2 - [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML1.ipynb)
        * Model persistence
        * Random Projection
        * SVM Kernal- Polynomial And RBF Implementation
        * Sigmoid function
        * K Nearest Neighbors
            * Decision Boundary for Knn
            * Effect of K on Decision Boundary
            * KNeighborsClassifier Hyper-parameters
            * Weighted Knn
            * KD Tree Algo
            * Ball Tree Algo
2. **Feature Engineering**
    * Standardization (Z-score Normalization) [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML2.ipynb)
    * Normalization [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML3.ipynb)
        * MinMax Scaling
        * Mean Normalization
        * Max Absolute Scaling
        * Robust Scaling
    * Encoding Categorical Data
        * Ordinal Encoding and Label Encoding [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML4.ipynb)
        * One Hot Encoding [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML5.ipynb)
    * Column Transformer [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML6.ipynb)
    * Pipeling
        * Training Titanic Dataset without using Pipeline [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML7.ipynb)
        * Predict Titanic Dataset from model.pkl without using Pipeline [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/ML8.ipynb)
    * Scikit learn Begineer Preprocessing [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/sci-kit_learn_preprocessing.ipynb)
    * Text Preprocessing [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/text-preprocessing.ipynb)
    * Text Representation [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/text-representation.ipynb)
3. **Accuracy Calculations** [Notebook](https://github.com/sukritiguin/ML/blob/main/Accuracy_Calculation.ipynb)
4. **Supervised Machine Learning Algorithms**
      - **Linear Model**
           * Implementation of `Simple Linear Regression` [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementation_Simple_Linear_Regression.ipynb)
           * Implementation of `Multiple Linear Regression` [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementation_Multiple_Linear_Regression.ipynb)
           * **`Ridge Regularization`** on **Simple Linear Regression** **[Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementing_Ridge_Regression_for_Simple_Linear_Regression.ipynb)**
           * **`Ridge Regularization`** on **Multiple Linear Regression** **[Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementing_Ridge_Regressor_for_Multiple_Linear_Regression.ipynb)**
           * Implementation of `Logistic Regression` for `Binary Classification` [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementation_Logistic_Regression.ipynb)
      - **K-Nearest Neighbors (KNN) algorithm**
           * Idea behind KNN [Idea](https://github.com/sukritiguin/ML/blob/main/Datasets/Idea%20Behind%20KNN.md)
           * Implementing KNN from scratch [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementing_KNN.ipynb)
         #### `sklearn.neighbors.KNeighborsClassifier` [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Hands_on_KNeighborsClassifier.ipynb)
         
         - Used for KNN classification.
         - Parameters include `n_neighbors` (number of neighbors), `weights` (weighting strategy), and `metric` (distance metric).
         - Provides methods for fitting the model (`fit`), making predictions (`predict`), and calculating class probabilities (`predict_proba`).
         - Find the best value of k
              * Iterative way
              * GridSearchCV
              * RandomizedSearchCV
              * Elbow Method
        - Hyperparameters Tuning
         
         #### `sklearn.neighbors.KNeighborsRegressor` [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Hands_on_KNeighborsRegressor.ipynb)
         
         - Used for KNN regression.
         - Parameters are similar to `KNeighborsClassifier`.
         - Provides methods for fitting the model (`fit`) and making predictions (`predict`).
         
         #### `sklearn.neighbors.NearestNeighbors`
         
         - Performs unsupervised nearest neighbors search.
         - Useful for finding k-nearest neighbors without associated labels.
         - Parameters include `n_neighbors`, `algorithm`, and `metric`.
         - Provides methods for fitting the model (`fit`), querying neighbors (`kneighbors`), and finding distances (`kneighbors_graph`).
         
         #### `sklearn.neighbors.RadiusNeighborsClassifier`
         
         - Performs classification based on neighbors within a specified radius.
         - Parameters include `radius`, `weights`, and `outlier_label`.
         - Provides methods for fitting the model (`fit`) and making predictions (`predict`).
         
         #### `sklearn.neighbors.RadiusNeighborsRegressor`
         
         - Performs regression based on neighbors within a specified radius.
         - Parameters are similar to `RadiusNeighborsClassifier`.
         - Provides methods for fitting the model (`fit`) and making predictions (`predict`).
         
         #### `sklearn.neighbors.KernelDensity`
         
         - Estimates the probability density function using kernel density estimation.
         - Useful for density-based clustering and visualization.
         - Parameters include `bandwidth` (kernel bandwidth) and `kernel` (kernel function).
      - **SVM(Support Vector Machine)**
           * Idea behind Hard Margin SVM [ReadMe](https://github.com/sukritiguin/ML/blob/main/Datasets/Implementing%20SVM%20Hard%20Margin.md)
           * Implementing SVM [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementing_Hard_Margin_Support_Vector_Machine_for_Binary_Classification.ipynb)
           * Hands on SVM [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Hands_on_SVM.ipynb)

6. **Unsupervised Machine Learning Algorithms**
     - **K Mean Clustering** [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/All_about_clustering.ipynb)
          * Implementing K Mean Clustering from scratch [Notebook](https://github.com/sukritiguin/ML/blob/main/NoteBooks/Implementing_K_Mean_Clustering.ipynb)
          * K Mean Clustering [2D, 3D]
          * Elbow Method & WCSS Calculations
          * Silhouette analysis
          * Agglomerative Clustering [2D, 3D]
          * DBSCAN [2D, 3D]
