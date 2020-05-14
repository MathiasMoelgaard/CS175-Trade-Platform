from agent_thread import agent_thread
from util import *
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from sklearn import preprocessing

#From https://github.com/philipperemy/keras-tcn
from tcn import TCN, tcn_full_summary

class tcn_agent(agent_thread):

    '''
    TCN network takes in the number of moments to look at

    '''
    def __init__(self, moments = 30, batch_size = None, input_dim = 1 ):
        agent_thread.__init__(self)
        self.moments = moments #Number of moments looked back to decide next value
        self.holding_time = 0
        self.batch_size = batch_size
        self.input_dim = input_dim #Dimensions of input default to 1
        self.built = False
        self.model() #Compile model
        self.training = True

    def _find_decision(self):
        inputs = list()
        if len(self.market_history) > self.moments:
            self.run_model(training=self.training)
        else:
            pass

    def model(self):
        #build model around TCN
        self.built = True
        i = Input(batch_shape=(self.batch_size, self.moments-1, self.input_dim))
        o = TCN(return_sequences=True)(i)
        o = TCN(return_sequences=False)(o)
        o = Dense(1)(o)
        self.m = Model(inputs=i, outputs=o)
        self.m.compile(optimizer='adam', loss='mse') #optimizer and loss can be changed to what we want

    def split_data(self, input, moments):
        # Split data into groups for training and testing
        x = np.array([])
        y = np.array([])
        input = [i.price for i in input]
        input = np.array(input)
        input = np.atleast_2d(input)
        #Normalize data
        input = self.normalization(input)
        for i in range(input.shape[1] - moments):
            x_values = np.array(input[0][i:moments+i])
            y_value = np.array(input[0][i+moments])
            if (x.shape[0] == 0):
                x = x_values
                y = [y_value]
            else:
                x = np.vstack((x, x_values))
                y = np.vstack((y, [y_value]))
        return x, y

    def prep_data(self):
        #Prepare training data here, need data to train on
        train_x, train_y = self.split_data(input, self.moments)

    def normalization(self, data, mode = 'default'):
        #To be added to with normalization methods
        if mode == 'default':
            return preprocessing.normalize(data)

    def run_model(self, training = False):
        #Currently it is made to run one behind so we can see real value
        #while predicting it
        inputs = self.market_history[-self.moments:]
        if training == True:
            #Use preloaded data
            #inputs = preloaded data
            #x, y = self.split_data(inputs, self.moments - 1)
            x, y = self.split_data(inputs, self.moments - 1)
            x = np.array(x)
            x = np.atleast_3d(x)
            y = np.atleast_2d(y)
            self.m.fit(x, y, epochs=10, validation_split=0.2)
        else:
            inputs = self.market_history[-self.moments:]
            x, y = self.split_data(inputs, self.moments - 1)
            x = np.array(x)
            x = np.atleast_3d(x)
            y = np.atleast_2d(y)
            y_hat = self.m.predict(x)
            print("Predicting next price to be: ", y_hat)
            print("Real next price was: ", y)