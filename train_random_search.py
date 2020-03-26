
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from datetime import timedelta, datetime

from data import data
from models.SIRC import SIRC
from models.SEIR import SEIR

# ==============================================================================
#
# Constants
#
# ==============================================================================

LOSS_CV_ROUNDS = 3

# ==============================================================================
#
# Functions
#
# ==============================================================================

def mse_loss(y_true, y_pred):
    mse = (np.square(y_true - y_pred)).mean()
    return mse


def mae_loss(y_true, y_pred):
    mae = (np.absolute(y_true - y_pred)).mean()
    return mae



def main(args):
    """ Train and evaluate with a model in a NON TIME SERIES form
    """

    # ==========================================================================
    # Get the dataset
    # ==========================================================================
    X, Y, dates = data.getDataset(init_population = args.population,
                           type=args.model.upper(),
                           country=args.train_on_country,
                           return_time_series= False,
                        )

    if not np.sum(X > 1) == 0:
        raise ValueError("Data is expected normalized in [0, 1]. \
                          The selected population is not enough.")
    
    # Select the model
    if args.model.upper() == 'SIRC':
        model = SIRC()
    elif args.model.upper() == 'SEIR':
        model = SEIR()
    else:
        raise ValueError('--model not supported')

    # Select the loss
    if args.loss.upper() == 'MSE':
        loss = mse_loss
    elif args.loss.upper() == 'MAE':
        loss = mae_loss
    else:
        raise ValueError('--loss not supported')

    print("{:>15} {:>15} {:>15} {:>15} {:>15} {:>15}  {:>15}  {:>15}".format(
        'iter','beta_mu','beta_rho','gamma_mu','gamma_rho','delta_mu','delta_rho','loss'))

    # Start from last checkpoint if exists
    old_data = os.path.isfile(args.cache_file)
    f = open(args.cache_file, 'a')
    if not old_data:
        f.write("iter,beta_mu,beta_rho,gamma_mu,gamma_rho,delta_mu,delta_rho,loss\n")
        optimal_params = None
        optimal_loss = np.inf
    else:
        logs = open(args.cache_file, 'r') .read().splitlines()
        last_log = logs[-1]
        last_log = last_log.split(',')
        optimal_params = {
            'beta_mu' : float(last_log[1]),
            'beta_rho' : float(last_log[2]),
            'gamma_mu' : float(last_log[3]),
            'gamma_rho' : float(last_log[4]),
            'delta_mu' : float(last_log[5]),
            'delta_rho' : float(last_log[6]),
            }
        optimal_loss = float(last_log[7])
    
        print("{: 15d} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E}".format(
                    -1, 
                    optimal_params['beta_mu'],
                    optimal_params['beta_rho'],
                    optimal_params['gamma_mu'], 
                    optimal_params['gamma_rho'], 
                    optimal_params['delta_mu'],
                    optimal_params['delta_rho'],optimal_loss))

    # ==========================================================================
    # Train by random search
    # ==========================================================================


    y_true = Y
    timesteps = 1

    def compute_loss(params={}, X = [], y_true = [], timesteps = 1):
        # Set params (random if value is not set)
        model.set_params(**params)

        # Determine the loss with those params
        l = 0
        for _ in range(LOSS_CV_ROUNDS):
            y_pred = model.predict(X[:,0], X[:,1], X[:,2], X[:,3], timesteps)
            y_pred = np.squeeze(y_pred, axis=1) # remove timestep dimension

            l += mae_loss(y_true, y_pred)
        l /= LOSS_CV_ROUNDS
        return l
        
    
    for it in range(args.train_iters):
        if it % 1000 == 0:
            print('Iter:', it)

        l = compute_loss(X = X, y_true = y_true, timesteps = 1)
        

        params = model.get_params()
        # TODO: implement occam razor for solutions! if (opt_loss - loss) < epsilon: 
        # chose the solution with higher variance
        if l < optimal_loss:
            optimal_params = params
            optimal_loss = l

            mess = "{},{},{},{},{},{},{},{}".format(
                    it,
                    params['beta_mu'],
                    params['beta_rho'],
                    params['gamma_mu'], 
                    params['gamma_rho'],
                    params['delta_mu'],
                    params['delta_rho'], l)
            f.write(mess+"\n")
            print("{: 15d} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E}".format(
                it, 
                params['beta_mu'],
                params['beta_rho'],
                params['gamma_mu'], 
                params['gamma_rho'], 
                params['delta_mu'],
                params['delta_rho'],l))

    # ==========================================================================
    # Train by scipy minimize
    # ==========================================================================

    # from scipy import optimize

    # def compute_loss(params, X = [], y_true = [], timesteps = 1):
    #     # Set params (random if value is not set)
    #     params = {'beta_mu' : params[0],
    #             'beta_rho' : params[1],
    #             'gamma_mu' : params[2],
    #             'gamma_rho' : params[3],
    #             'delta_mu' : params[4],
    #             'delta_rho' : params[5],}

    #     model.set_params(**params)

    #     # Determine the loss with those params
    #     l = 0
    #     for _ in range(LOSS_CV_ROUNDS):
    #         y_pred = model.predict(X[:,0], X[:,1], X[:,2], X[:,3], timesteps)
    #         y_pred = np.squeeze(y_pred, axis=1) # remove timestep dimension

    #         l += mae_loss(y_true, y_pred)
    #     l /= LOSS_CV_ROUNDS
    #     return l

    # res =optimize.minimize(fun = compute_loss, 
    #                     #    x0 = np.array([0.38, 0.02, 30., 7., 0.17, 0.02]),
    #                        x0 = np.array([0.8, 0.1, 30., 1., 0.1, 0.01]),
    #                        args = (X, y_true, 1),
    #                     #    method = "L-BFGS-B",
    #                        method = "SLSQP",
    #                        # Min / max bound for beta_mu, beta_rho, ....
    #                        bounds = [(0, 5), (0, 5), (0, 40), (0, 10), (0,1), (0,1)],
    #                        tol = 1e-6,
    #                        options = {'maxiter':100000, 'disp':True}
    #                        )

        


    # ==========================================================================
    # Evaluate on Italy
    # ==========================================================================
    def reject_outliers(data, m=1.):
        """ Sets the value of outlayers to nan
        Args:
            data (np.array): the data from which to remove outlayers
            m (float): threshold on outlayers as m * std_dev(data)
        """

        standardized_data = np.abs(data - np.mean(data, axis=0))
        outlayer_threshold = m * np.std(data, axis=0)
        outlayers = standardized_data > outlayer_threshold

        # data[ outlayers ] = standardized_data[outlayers]
        data[ outlayers ] = np.nan

        print('nr of outlayers: ', np.sum(outlayers))
        return data

    model.set_params(beta_mu=optimal_params['beta_mu'],
                     gamma_mu=optimal_params['gamma_mu'],
                     beta_rho=optimal_params['beta_rho'],
                     gamma_rho=optimal_params['gamma_rho'])

    COUNTRY = 'Italy'
    POPULATION = args.population

    X, _, _ = data.getDataset(
                           init_population = POPULATION,
                           type=args.model.upper(), country=COUNTRY)
    X = X[0,:,:] # first sample
    # get the last day
    S = X[-1, 0]
    I = X[-1, 1]
    R = X[-1, 2]
    C = X[-1, 3]


    # Average more predictions (models have bayesian params)
    y_pred_accumulator = []
    for _ in range(1000):
        y_pred = model.predict(S, I, R, C, timesteps=args.predicted_days)
        y_pred_accumulator.append(y_pred)

    y_pred_accumulator = np.array(y_pred_accumulator)
    y_pred_accumulator = reject_outliers(y_pred_accumulator)

    # Max, Min, Mean ignoring NaNs (--> outlayers)
    worst_case = np.nanmax(y_pred_accumulator, axis=0)
    best_case = np.nanmin(y_pred_accumulator, axis=0)
    average_case = np.nanmean(y_pred_accumulator, axis=0)

    timeseries_best = np.concatenate([X, best_case], axis=0)
    timeseries_worst = np.concatenate([X, worst_case], axis=0)
    timeseries_average = np.concatenate([X, average_case], axis=0)


    # Adding the population to the values
    timeseries_best *= args.population
    timeseries_worst *= args.population
    timeseries_average *= args.population


    # add predicted values to dates
    current = dates[-1]
    for i in range(args.predicted_days):
        dates.append( current + timedelta(days = i+1))
    dates = [d.strftime('%d/%m/%y') for d in dates]

    plt.figure(figsize=(15,10))
    plt.title('Italy - COVID-19')

    # Plot average
    plt.plot(timeseries_average[:,1], label='Infected',  color='orange')
    plt.plot(timeseries_average[:,2], label='Recovered',  color='g')
    plt.plot(timeseries_average[:,3], label='Casualties',  color='r')


    # "Confidence interval"
    x = np.arange(0, len(timeseries_best)) 
    plt.fill_between(x, timeseries_best[:,1], timeseries_worst[:,1], color='orange', alpha=0.2)
    plt.fill_between(x, timeseries_best[:,2], timeseries_worst[:,2], color='g', alpha=0.2)
    plt.fill_between(x, timeseries_best[:,3], timeseries_worst[:,3], color='r', alpha=0.2)
    
    plt.xticks(x[::5], dates[::5], rotation=90)
    # plt.ylim([-1, args.population+10])
    plt.grid(True)
    plt.legend()
    plt.plot()
    plt.savefig("Italy.png")
    plt.show()

    
        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "COVID-19 estimation of params")
    parser.add_argument('--loss', type=str, default='MAE', help='The loss type (MSE or MAE) [default: MSE]')
    parser.add_argument('--model', type=str, default='SIRC', help='The loss type (SIRC or SEIR) [default: SIR]')
    parser.add_argument('--train_on_country', type=str, default=None, help='The country to use to train the model \
        (None: all the contries) [default: None]')
    parser.add_argument('--population', type=int, default=100000, help='Population [default: 100k]')
    parser.add_argument('--train_iters', type=int, default=100000, help='Train iters [default: 100k]')
    parser.add_argument('--predicted_days', type=int, default=15, help='Days to predict [default: 15]')
    parser.add_argument('--cache_file', type=str, default='optimization_steps.csv', help='The cache where to save optimal params')

    args = parser.parse_args()    

    main(args)
