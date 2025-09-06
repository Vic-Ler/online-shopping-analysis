# run_predictions.py
import pandas as pd

import pandas as pd
from steps.prediction_loader import predict_from_raw_row


if __name__ == "__main__":
    # Example input row
    raw_test_row = pd.DataFrame({
        "Administrative": [1],
        "Administrative_Duration": [0],
        "Informational": [0],
        "Informational_Duration": [0],
        "ProductRelated": [20],
        "ProductRelated_Duration": [200],
        "BounceRates": [0],
        "ExitRates": [0],
        "PageValues": [0],
        "SpecialDay": [0],
        "Month": ["Feb"],
        "VisitorType": ["Returning_Visitor"],
        "Weekend": [0],
    })

    prediction, probabilities = predict_from_raw_row(raw_test_row)

    print("Prediction:", prediction)
    if probabilities is not None:
        print("Probabilities:", probabilities)
