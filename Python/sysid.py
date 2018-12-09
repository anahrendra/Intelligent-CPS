import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

HIDDEN_LAYERS   = 20
HIDDEN_NEURONS  = 64
NUM_EPOCHS      = 100


def read_from_file(fname):
    file = open(fname,"r")
    y = []
    t = []
        
    for line in file:
        line_s = line.split("\n")
        line_spl = line_s[0].split(',')
        t.append(float(line_spl[1]))
        y.append(float(line_spl[2]))
    file.close()

    return t,y

if __name__ == "__main__":
    X,Y = read_from_file("plant.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(64, activation = 'relu', input_dim = 1))

    # Adding the hidden layers
    for i in range(HIDDEN_LAYERS):
        model.add(Dense(units = HIDDEN_NEURONS, activation = 'relu'))

    # Adding the output layer
    model.add(Dense(units = 1))

    #model.add(Dense(1))
    # Compiling the ANN
    model.compile(optimizer = 'nadam', loss = 'mean_squared_error')

    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size = 10, epochs = NUM_EPOCHS)

    
    y_pred = model.predict(X)
    print y_pred

    plt.plot(X,Y, color = 'red', label = 'Real data')
    plt.plot(X,y_pred, color = 'blue', label = 'Predicted data')
    plt.title('Prediction')
    plt.legend()
    plt.show()