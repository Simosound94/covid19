
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from datetime import timedelta, datetime
import pickle as pkl

from data import data
from predict_utils import predict
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


    # cache file
    if os.path.isfile(args.cache_file):
        print('Loading from cache')
        optimal_params = pkl.load(open(args.cache_file, 'rb'))
    else:
        optimal_params = {'beta_mu': [], 'beta_rho': [], 'gamma_mu':[],
        'gamma_rho':[], 'delta_mu':[], 'delta_rho':[], 'predicted_dates':[]} 

    # ==========================================================================
    # Estimate parameters day by day
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


    print("{:^20} {:>20} {:>20} {:>20} {:>20}".format(
        'date','beta','gamma','delta','loss'))
    for t in range(args.days_to_compute_params, len(X)):
        date = dates[t -(int) (args.days_to_compute_params/2)].strftime("%d/%m/%Y")

        # If element in cache file, skip it
        if date in optimal_params['predicted_dates']:
            i = optimal_params['predicted_dates'].index(date)
            print("{:^20} {: 15.2E}±{:.2E} {: 15.2E}±{:.2E} {: 15.2E}±{:.2E} {}".format(
                optimal_params['predicted_dates'][i],
                optimal_params['beta_mu'][i],
                optimal_params['beta_rho'][i],
                optimal_params['gamma_mu'][i], 
                optimal_params['gamma_rho'][i], 
                optimal_params['delta_mu'][i],
                optimal_params['delta_rho'][i], '[from cache]'))
            continue
        
        # Initialize optimal params
        for param_name in optimal_params:
            optimal_params[param_name].append(None)
        optimal_params['predicted_dates'][-1] = date
        optimal_loss = np.inf

        # Data as the data of those days
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

        print("{:^20} {: 15.2E}±{:.2E} {: 15.2E}±{:.2E} {: 15.2E}±{:.2E} {: 15.2E}".format(
                optimal_params['predicted_dates'][-1],
                params['beta_mu'],
                params['beta_rho'],
                params['gamma_mu'], 
                params['gamma_rho'], 
                params['delta_mu'],
                params['delta_rho'],l))


    # save cache file
    pkl.dump(optimal_params, open(args.cache_file, 'wb'))

    # ==========================================================================
    # Fit linear regression on model parameters and predict what's next
    # ==========================================================================

    optimal_params = {k: np.array(v) for k,v in optimal_params.items()}
    predicted_dates = optimal_params['predicted_dates'].copy()
    del optimal_params['predicted_dates']

    # smooth average
    def smooth(x):
        window = 3
        last = x[-window+1:]
        averaged = np.convolve(x, np.ones((window,))/window, mode='same')
        averaged[-window+1:] = last # do not average last samples (avoid distortions)
        return averaged

    optimal_params = {k: smooth(v) for k,v in optimal_params.items()}


    # fit Beta (that changes according to social distancing etc)
    # linear regression on the last computed params
    DAYS_TO_INFER_PARAMS = 10
    inferred_coeffs = np.polyfit(x = np.arange(0, DAYS_TO_INFER_PARAMS),
                   y = optimal_params['beta_mu'][-DAYS_TO_INFER_PARAMS:],
                   deg = 1)
 
    x = np.arange(0, args.predicted_days)
    beta_mu = x*inferred_coeffs[0] + inferred_coeffs[1]
    beta_mu = np.clip(beta_mu, 0, np.inf)

    # Other parms are considered stables, we consider the mean
    beta_rho = np.mean(optimal_params['beta_rho'])
    gamma_mu = np.mean(optimal_params['gamma_mu'])
    gamma_rho = np.mean(optimal_params['gamma_rho'])
    delta_mu = np.mean(optimal_params['delta_mu'])
    delta_rho = np.mean(optimal_params['delta_rho'])
    
    beta_rho = np.array([ beta_rho]*args.predicted_days) 
    gamma_mu = np.array([ gamma_mu]*args.predicted_days) 
    gamma_rho= np.array([gamma_rho]*args.predicted_days)
    delta_mu = np.array([ delta_mu]*args.predicted_days) 
    delta_rho= np.array([delta_rho]*args.predicted_days)

    params_next_days = []
    for i in range(args.predicted_days):
        params_next_days.append({'beta_mu': beta_mu[i], 'beta_rho': beta_rho[i],
                                 'gamma_mu':gamma_mu[i],  'gamma_rho':gamma_rho[i],
                                 'delta_mu':delta_mu[i], 'delta_rho':delta_rho[i]})


    predict(model, args.train_on_country, args.population,
        args.predicted_days, params_next_days, std_dev=0.1)



    # ==========================================================================
    # Compute R0 and show params
    # ==========================================================================

    # add predicted future params
    current = datetime.strptime(predicted_dates[-1], '%d/%m/%Y')
    dates = []
    for i in range(args.predicted_days):
        dates.append( current + timedelta(days = i+1))
    dates = np.array([d.strftime('%d/%m/%y') for d in dates])
    predicted_dates = np.concatenate((predicted_dates, dates))

    optimal_params['beta_mu'] = np.concatenate((optimal_params['beta_mu'], beta_mu))
    optimal_params['beta_rho'] = np.concatenate((optimal_params['beta_rho'], beta_rho))
    optimal_params['gamma_mu'] = np.concatenate((optimal_params['gamma_mu'], gamma_mu))
    optimal_params['gamma_rho'] = np.concatenate((optimal_params['gamma_rho'], gamma_rho))
    optimal_params['delta_mu'] = np.concatenate((optimal_params['delta_mu'], delta_mu))
    optimal_params['delta_rho'] = np.concatenate((optimal_params['delta_rho'], delta_rho))

    # compute R0
    optimal_params['R0_mu'] = optimal_params['beta_mu']*optimal_params['gamma_mu']
    optimal_params['R0_rho'] = optimal_params['beta_rho']*optimal_params['gamma_rho']
    

    fig, axs = plt.subplots(4, sharex=True, figsize=(15,15))
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
    parser.add_argument('--predicted_days', type=int, default=15, help='Days to predict [default: 15]')
    parser.add_argument('--cache_file', type=str, default='optimal_params.pkl', help='The cache where to save optimal params')


    args = parser.parse_args()    

    main(args)
