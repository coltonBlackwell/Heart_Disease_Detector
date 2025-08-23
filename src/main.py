from utils.train_model import train
from utils.test_model import test

def main():

    X, model, scaler, y_test, y_pred= train(input_csv='../data/processed.cleveland.data') # Training the model

    test(X, model, scaler, y_test, y_pred) # Testing the model

    # Using the model for user input and heart disease prediction


if __name__ == '__main__':
    main()
