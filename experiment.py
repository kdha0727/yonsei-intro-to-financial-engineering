import time
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import scipy
from statsmodels.stats.stattools import durbin_watson


def bscall(S, K, T, r, sig):
    d1 = (np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = (np.log(S/K)+(r-0.5*sig**2)*T)/(sig*np.sqrt(T))
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)


def bsput(S, K, T, r, sig):
    d1 = (np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = (np.log(S/K)+(r-0.5*sig**2)*T)/(sig*np.sqrt(T))
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


def initialize_model(N, payoff_lambda, method=None):

    if method == 'dense':
        method = None

    if method is None:

        def delta_model_func(price, name):
            model = tf.keras.Sequential(name=name)
            model.add(tf.keras.layers.Dense(32, activation='tanh', name="dense1"))
            model.add(tf.keras.layers.BatchNormalization(name='bn1'))
            model.add(tf.keras.layers.Dense(32, activation='tanh', name="dense2"))
            model.add(tf.keras.layers.BatchNormalization(name='bn2'))
            model.add(tf.keras.layers.Dense(32, activation='tanh', name="dense3"))
            model.add(tf.keras.layers.BatchNormalization(name='bn3'))
            model.add(tf.keras.layers.Dense(16, activation='tanh', name="dense4"))
            model.add(tf.keras.layers.Dense(1, name="dense5"))
            return model(price)

    else:
        assert isinstance(method, str) and method.lower() in ('rnn', 'gru', 'lstm')
        layer = dict(
            rnn=tf.keras.layers.SimpleRNN,
            gru=tf.keras.layers.GRU,
            lstm=tf.keras.layers.LSTM,
        )[method.lower()]

        lstm_1 = layer(32, return_state=True, activation='tanh', name=f'{method}1')
        bn_1 = tf.keras.layers.BatchNormalization(name='bn1')
        lstm_2 = layer(32, return_state=True, activation='tanh', name=f'{method}2')
        bn_2 = tf.keras.layers.BatchNormalization(name='bn2')
        lstm_3 = layer(32, return_state=True, activation='tanh', name=f'{method}3')
        bn_3 = tf.keras.layers.BatchNormalization(name='bn3')
        reshape_input = tf.keras.layers.Reshape((1, 1), input_shape=(1,), name='reshape_input')
        reshape_output = tf.keras.layers.Reshape((1, 32), input_shape=(32,), name='reshape_output')
        lstm_state_1 = lstm_state_2 = lstm_state_3 = None

        def delta_model_func(price, name):
            nonlocal lstm_state_1, lstm_state_2, lstm_state_3
            delta, *lstm_state_1 = lstm_1(reshape_input(price), initial_state=lstm_state_1)
            delta = bn_1(delta)
            delta, *lstm_state_2 = lstm_2(reshape_output(delta), initial_state=lstm_state_2)
            delta = bn_2(delta)
            delta, *lstm_state_3 = lstm_3(reshape_output(delta), initial_state=lstm_state_3)
            delta = bn_3(delta)
            delta = tf.keras.layers.Dense(16, name='%s_dense_1' % name, activation='tanh')(delta)
            delta = tf.keras.layers.Dense(1, name='%s_dense_2' % name)(delta)
            return delta

    price = tf.keras.layers.Input(shape=(1,), name="price")

    inputs = [price]

    hedge_cost = tf.keras.layers.Lambda(lambda x: x*0.0)(price)

    for j in range(N):
        delta = delta_model_func(price, name='delta_model_%s' % j)
        new_price = tf.keras.layers.Input(shape=(1,), name='S%s' % j)
        inputs.append(new_price)
        price_inc = tf.keras.layers.Subtract(name='price_inc_%s' % j)([price, new_price])
        cost = tf.keras.layers.Multiply(name="multiply_%s" % j)([delta, price_inc])
        hedge_cost = tf.keras.layers.Add(name='cost_%s' % j)([hedge_cost, cost])
        price = new_price

    payoff = tf.keras.layers.Lambda(payoff_lambda, name='payoff')(price)
    cum_cost = tf.keras.layers.Add(name="hedge_cost_plus_payoff")([hedge_cost, payoff])

    return tf.keras.Model(inputs=inputs, outputs=cum_cost)


def make_entry(T, r, sig, M):

    entry = {}
    S0 = 1.
    K_1 = 0.9; K_2 = 0.95; K_3 = 1.0; K_4 = 1.05; K_5 = 1.1

    option_name = "0-Covered-Call"
    premium = (
        - bsput(S0,K_3,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        - tf.math.maximum(K_3-x,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "1-Married-Put"
    premium = (
        bscall(S0,K_3,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(x-K_3,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "2-Bull-Call-Spread"
    premium = (
        bscall(S0,K_1,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_5,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(x-K_1,0)
        - tf.math.maximum(x-K_5,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "3-Bear-Put-Spread"
    premium = (
        - bsput(S0,K_1,T,r,sig) * np.ones([M,1])
        + bsput(S0,K_5,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        - tf.math.maximum(K_1-x,0)
        + tf.math.maximum(K_5-x,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "4-Protective-Collar"
    premium = (
        bscall(S0,K_2,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_4,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(x-K_2,0)
        - tf.math.maximum(x-K_4,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "5-Long-Straddle"
    premium = (
        + bsput(S0,K_3,T,r,sig) * np.ones([M,1])
        + bscall(S0,K_3,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(K_3-x,0)
        + tf.math.maximum(x-K_3,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "6-Long-Strangle"
    premium = (
        + bsput(S0,K_1,T,r,sig) * np.ones([M,1])
        + bscall(S0,K_5,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(K_1-x,0)
        + tf.math.maximum(x-K_5,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "7-Long-Call-Butterfly-Spread"
    premium = (
        + bscall(S0,K_1,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_3,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_3,T,r,sig) * np.ones([M,1])
        + bscall(S0,K_5,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(x-K_1,0)
        - tf.math.maximum(x-K_3,0)
        - tf.math.maximum(x-K_3,0)
        + tf.math.maximum(x-K_5,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "8-Iron-Condor"
    premium = (
        bscall(S0,K_1,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_2,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_4,T,r,sig) * np.ones([M,1])
        + bscall(S0,K_5,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(x-K_1,0)
        - tf.math.maximum(x-K_2,0)
        - tf.math.maximum(x-K_4,0)
        + tf.math.maximum(x-K_5,0)
    )
    entry[option_name] = premium, payoff_lambda

    option_name = "9-Iron-Butterfly"
    premium = (
        + bsput(S0,K_1,T,r,sig) * np.ones([M,1])
        - bsput(S0,K_3,T,r,sig) * np.ones([M,1])
        - bscall(S0,K_3,T,r,sig) * np.ones([M,1])
        + bscall(S0,K_5,T,r,sig) * np.ones([M,1])
    )
    payoff_lambda = lambda x : (
        tf.math.maximum(K_1-x,0)
        - tf.math.maximum(K_3-x,0)
        - tf.math.maximum(x-K_3,0)
        + tf.math.maximum(x-K_5,0)
    )
    entry[option_name] = premium, payoff_lambda

    return entry


if __name__ == '__main__':

    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Set Other Values
    S0=1.; r=.0; sig=.15; T=30/365; M=1000; N=30
    dt = T/N; rdt = r*dt; sigsdt = sig * np.sqrt(dt)

    # Prepare Values
    S = np.empty([M, N+1])
    rv = np.random.normal(r*dt,sigsdt,[M,N])

    for i in range(M):
        S[i,0] = S0
        for j in range(N):
            S[i,j+1] = S[i,j] * (1+rv[i,j])

    with open("result.csv", "w") as csv:

        csv.write("option_name,name,infer_time,mse,mae,normstatistic,normpvalue,independency,\n")

        for option_name, (premium, payoff_lambda) in make_entry(T, r, sig, M).items():

            SS = [S[:,i].reshape(M,1) for i in range(N+1)]
            x = [SS]
            y = np.ones([M,1]) * premium

            for name in ['dense', 'rnn', 'gru', 'lstm']:
                model = initialize_model(N, payoff_lambda=payoff_lambda, method=name)
                model.compile(loss='mse', optimizer='adam')
                csv_logger = tf.keras.callbacks.CSVLogger(
                    f'history-{option_name}-{name}.csv', append=True, separator=';'
                )
                history = model.fit(x, y,
                                    batch_size=32, epochs=50, callbacks=[csv_logger], verbose=False)
                infer_time = time.time()
                prediction = model.predict(x)
                infer_time = time.time() - infer_time

                residuals = prediction - y
                mse = (residuals ** 2).mean()
                mae = abs(residuals).mean()

                normstatistic, normpvalue = scipy.stats.shapiro(residuals)

                sorted_residuals = [
                    el[1] for el in 
                    sorted(zip(S[:, -1], residuals.reshape(-1)), key=lambda el: el[0])
                ]
                independency = durbin_watson(sorted_residuals)

                csv.write(
                    f"{option_name},{name},{infer_time},"
                    f"{mse},{mae},{normstatistic},{normpvalue},{independency},\n"
                )
