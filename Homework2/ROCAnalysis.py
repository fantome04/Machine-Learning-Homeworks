class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_pred = y_predicted
        self.y_true = y_true
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
        for i in range(len(y_predicted)):
            if y_predicted[i] == 1 and y_true[i] == 1:
                self.tp += 1
            elif y_predicted[i] == 0 and y_true[i] == 0:
                self.tn += 1
            elif y_predicted[i] == 1 and y_true[i] == 0:
                self.fp += 1
            elif y_predicted[i] == 0 and y_true[i] == 1:
                self.fn += 1

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        if self.fp + self.tn == 0:
            return 0
        return self.fp / (self.fp + self.tn)
    
    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)
  
    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        precision = self.precision()
        recall = self.tp_rate()
        
        if precision + recall == 0:
            return 0
        
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)