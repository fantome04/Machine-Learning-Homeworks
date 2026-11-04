from abc import ABC, abstractmethod
import numpy as np

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

    def _polynomial_features(self, X):
        """
            Generate polynomial features from the input features.
            Check the slides for hints on how to implement this one. 
            This method is used by the regression models and must work
            for any degree polynomial
            Parameters:
            X (array-like): Features of the data.

            Returns:
            X_poly (array-like): Polynomial features.
        """ 
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X_poly = np.ones((X.shape[0], 1))
        
        for d in range(1, self.degree + 1):
            X_poly = np.column_stack([X_poly, X**d])
            
        return X_poly
        

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        self.degree = degree
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X = np.array(X)
        y = np.array(y)
        
        X_ext = self._polynomial_features(X)
        self.beta = np.linalg.inv(X_ext.T.dot(X_ext)).dot(X_ext.T).dot(y)
        
        j = np.dot(X_ext,self.beta) - y
        J = (j.T.dot(j)) / len(y)
        self.cost_history = J

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X = np.array(X)
        
        X_ext = self._polynomial_features(X)
        return X_ext.dot(self.beta)
        
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        X = np.array(X)
        y = np.array(y)
        
        X_ext = self._polynomial_features(X)
        
        n = len(y)
        predictions = X_ext.dot(self.beta)
        errors = predictions - y
            
        return (1 / n) * np.dot(errors.T, errors)
        
class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X = np.array(X)
        y = np.array(y)
        
        X_ext = self._polynomial_features(X)
        self.beta = np.zeros(X_ext.shape[1])
        
        n = len(y)
        
        for _ in range(self.num_iterations):
            predictions = X_ext.dot(self.beta)
            errors = predictions - y
            
            j = np.dot(X_ext,self.beta) - y
            J = (j.T.dot(j)) / n
            self.cost_history.append(J)
            
            gradient = (2 / n) * X_ext.T.dot(errors)
            self.beta -= self.learning_rate * gradient
        
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X = np.array(X)
        
        X_ext = self._polynomial_features(X)
        return X_ext.dot(self.beta)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        X = np.array(X)
        y = np.array(y)
        
        X_ext = self._polynomial_features(X)
        
        n = len(y)
        predictions = X_ext.dot(self.beta)
        errors = predictions - y
            
        return (1 / n) * np.dot(errors.T, errors)
        

class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_ext = np.column_stack((np.ones((X.shape[0], 1)), X))
        
        self.beta = np.zeros(X_ext.shape[1])
        
        n = len(y)
        
        for _ in range(self.num_iterations):
            g = self._sigmoid(X_ext.dot(self.beta))
            
            self.cost_history.append(self._cost_function(X_ext, y))
            
            gradient = (1 / n) * X_ext.T.dot(g - y)
            self.beta -= self.learning_rate * gradient
            
    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_ext = np.column_stack((np.ones((X.shape[0], 1)), X))
        probabilities = self._sigmoid(X_ext.dot(self.beta)) 
    
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_ext = np.column_stack((np.ones((X.shape[0], 1)), X))
        
        return self._cost_function(X_ext, y)

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        n = len(y)
        z = X.dot(self.beta)
        g = self._sigmoid(z)
        
        epsilon = 1e-15
        g = np.clip(g, epsilon, 1 - epsilon)
        
        cost = (-1 / n) * (y.T.dot(np.log(g)) + (1 - y).T.dot(np.log(1 - g))) 
        return cost
        
    
class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.shape[1] != 2:
            print("NonLinearLogisticRegression requires exactly 2 features")
            return
        
        X_ext = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        
        self.beta = np.zeros(X_ext.shape[1])
        
        n = len(y)
        
        for _ in range(self.num_iterations):
            g = self._sigmoid(X_ext.dot(self.beta))
            
            self.cost_history.append(self._cost_function(X_ext, y))
            
            gradient = (1 / n) * X_ext.T.dot(g - y)
            self.beta -= self.learning_rate * gradient

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        X = np.array(X)
        
        if X.shape[1] != 2:
            print("NonLinearLogisticRegression requires exactly 2 features")
            return
        
        X_ext = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        probabilities = self._sigmoid(X_ext.dot(self.beta)) 
    
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.shape[1] != 2:
            print("NonLinearLogisticRegression requires exactly 2 features")
            return
        
        X_ext = self.mapFeature(X[:, 0], X[:, 1], self.degree)
        
        return self._cost_function(X_ext, y)

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        one = np.ones((len(X1), 1))
        
        X_ext = np.column_stack((one, X1.reshape(-1, 1), X2.reshape(-1, 1)))
        
        for i in range(2, D + 1):
            for j in range(0, i + 1):
                X_new = (X1 ** (i - j)) * (X2 ** j)
                X_ext = np.append(X_ext, X_new.reshape(-1, 1), axis=1)
                
        return X_ext

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        n = len(y)
        z = X.dot(self.beta)
        g = self._sigmoid(z)
        
        epsilon = 1e-15
        g = np.clip(g, epsilon, 1 - epsilon)
        
        cost = (-1 / n) * (y.T.dot(np.log(g)) + (1 - y).T.dot(np.log(1 - g))) 
        return cost