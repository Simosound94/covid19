
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
    if args.train_on_country is None:
        raise ValueError("select a specific country with --train_on_country")

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

    
    # ==========================================================================
    # Train by random search
    # ==========================================================================

    def compute_loss(params={}, X = [], y_true = [], timesteps = 1):
        # Set params (random if value is not set)
        model.set_params(**params)

        # Determine the loss with those params
        l = 0
        for _ in range(LOSS_CV_ROUNDS):
            y_pred = model.predict(X_step[:,0], X_step[:,1], X_step[:,2], X_step[:,3], timesteps)
            y_pred = np.squeeze(y_pred, axis=1) # remove timestep dimension
            l += mae_loss(y_true, y_pred)
        l /= LOSS_CV_ROUNDS
        return l


    optimal_params = {'beta_mu': [], 'beta_rho': [], 'gamma_mu':[],
        'gamma_rho':[], 'delta_mu':[], 'delta_rho':[]} 
    predicted_dates = []

    print("{:^20} {:>20} {:>20} {:>20} {:>20}".format(
        'date','beta','gamma','delta','loss'))
    for t in range(args.days_to_compute_params, len(X)):
        
        for param_name in optimal_params:
            optimal_params[param_name].append(None)
        optimal_loss = np.inf

        X_step = X[t-args.days_to_compute_params:t]
        Y_step = Y[t-args.days_to_compute_params:t]
        timesteps = 1
            
        for it in range(args.train_iters):
            l = compute_loss(X = X_step, y_true = Y_step, timesteps = 1)
            params = model.get_params()

            # TODO: implement occam razor for solutions! if (opt_loss - loss) < epsilon: 
            # chose the solution with higher variance
            if l < optimal_loss:
                # Set last optimal param as the value current param
                for k,v in params.items():
                    optimal_params[k][-1] = v
                optimal_loss = l

        predicted_dates.append(
            dates[t -(int) (args.days_to_compute_params/2) ].strftime("%d/%m/%Y"))
        print("{:^20} {: 15.2E}±{:.2E} {: 15.2E}±{:.2E} {: 15.2E}±{:.2E} {: 15.2E}".format(
                predicted_dates[-1],
                params['beta_mu'],
                params['beta_rho'],
                params['gamma_mu'], 
                params['gamma_rho'], 
                params['delta_mu'],
                params['delta_rho'],l))

    optimal_params = {k: np.array(v) for k,v in optimal_params.items()}
    # smooth average
    def smooth(x):
        window = 3
        last = x[-window+1:]
        averaged = np.convolve(x, np.ones((window,))/window, mode='same')
        averaged[-window+1:] = last # do not average last samples (avoid distortions)
        return averaged

    optimal_params = {k: smooth(v) for k,v in optimal_params.items()}
    optimal_params['R0_mu'] = optimal_params['beta_mu']*optimal_params['gamma_mu']
    optimal_params['R0_rho'] = optimal_params['beta_rho']*optimal_params['gamma_rho']
    


    predicted_dates = np.array(predicted_dates)
    fig, axs = plt.subplots(4, sharex=True, figsize=(15,15))
    # fig.figure(figsize=(15,10))
    plt.title(args.train_on_country+' - Params')

    # Plot average
    axs[0].plot(optimal_params['R0_mu'], label='R0', color='b')
    axs[1].plot(optimal_params['beta_mu'], label='Beta', color='r')
    axs[2].plot(optimal_params['gamma_mu'], label='Gamma',  color='orange')
    axs[3].plot(optimal_params['delta_mu'], label='Delta',  color='g')


    # TODO: Set delta to NaN if no deaths / recoveries in that period


    # "Confidence interval"
    x = np.arange(0, len(predicted_dates))
    axs[0].fill_between(x, optimal_params['R0_mu']+optimal_params['R0_rho'],
        optimal_params['R0_mu']-optimal_params['R0_rho'],  color='b', alpha=0.2)
    axs[1].fill_between(x, optimal_params['beta_mu']+optimal_params['beta_rho'],
        optimal_params['beta_mu']-optimal_params['beta_rho'],  color='r', alpha=0.2)
    axs[2].fill_between(x, optimal_params['gamma_mu']+optimal_params['gamma_rho'],
        optimal_params['gamma_mu']-optimal_params['gamma_rho'],  color='orange', alpha=0.2)
    axs[3].fill_between(x, optimal_params['delta_mu']+optimal_params['delta_rho'],
        optimal_params['delta_mu']-optimal_params['delta_rho'],  color='g', alpha=0.2)
    
    plt.xticks(x[::5], predicted_dates[::5], rotation=45)
    for ax in axs:
        ax.grid(True)
        ax.legend()
    axs[0].set_ylim(0, 10)
    axs[1].set_ylim(0, 4)
    axs[2].set_ylim(5, 40)
    axs[3].set_ylim(0, 1)
    plt.plot()
    plt.savefig(args.train_on_country+"_params.png")
    plt.show()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "COVID-19 estimation of params")
    parser.add_argument('--loss', type=str, default='MAE', help='The loss type (MSE or MAE) [default: MAE]')
    parser.add_argument('--model', type=str, default='SIRC', help='The loss type (SIRC or SEIR) [default: SIRC]')
    parser.add_argument('--train_on_country', type=str, default=None, help='The country to use to train the model \
        (None: all the contries) [default: None]')
    parser.add_argument('--population', type=int, default=100000, help='Population [default: 100k]')
    parser.add_argument('--train_iters', type=int, default=10000, help='Train iters [default: 10k]')
    parser.add_argument('--days_to_compute_params', type=int, default=3, help='Number of days used to compute each param')

    args = parser.parse_args()    

    main(args)
