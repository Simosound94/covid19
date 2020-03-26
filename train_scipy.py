
# import os
# import numpy as np
# import pandas as pd
# from scipy.integrate import solve_ivp
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# from datetime import timedelta, datetime

# START_DATE = {
#   'Japan': '1/22/20',
#   'Italy': '1/31/20',
#   'Republic of Korea': '1/22/20',
#   'Iran (Islamic Republic of)': '2/19/20'
# }

# class Learner(object):
#     def __init__(self, country, dataset_path, population):
#         self.country = country
#         self.loss = self.get_loss_funtion()
#         self.dataset_path = dataset_path
#         self.population = population

#         # data
#         self.data = self.load_confirmed(self.country)

#         # init params
#         self._beta = 0.00001713
#         self._gamma = 0.01414756

#     def get_init_point(self):
#         """ Returns the init point of Suceptibles, Infectuos, Recovered
#             [S_0, I_0, R_0]
#         """
#         assert self.data is not None
#         return [self.population, self.data[0], 0]


#     def load_confirmed(self, country):
#       """
#       Load confirmed cases downloaded from HDX
#       """
#       dataset_path = os.path.join(self.dataset_path, 
#                                   'time_series_19-covid-Confirmed.csv')

#       df = pd.read_csv(dataset_path)
#       country_df = df[df['Country/Region'] == country]
#       self.data = country_df.iloc[0].loc[START_DATE[country]:]
#       return self.data

#     def extend_index(self, index, new_size, date_format = '%m/%d/%y'):
#         values = index.values.tolist()
#         current = datetime.strptime(index[-1], date_format)
#         while len(values) < new_size:
#             current = current + timedelta(days=1)
#             values.append(datetime.strftime(current, date_format))

#         values = np.array([datetime.strptime(v, date_format).strftime('%d/%m/%y') for v in values])
#         return values

#     def _predict(self, beta, gamma, data):
#         """
#         Predict how the number of people in each compartment can be changed through time toward the future.
#         The model is formulated with the given beta and gamma.
#         """
#         predict_range = 150
#         new_index = self.extend_index(data.index, predict_range)
#         size = len(new_index)
#         def SIR(t, y):
#             S = y[0]
#             I = y[1]
#             R = y[2]
#             return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
#         extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
#         return new_index, extended_actual, solve_ivp(SIR, [0, size], self.get_init_point(), t_eval=np.arange(0, size, 1))

#     def get_loss_funtion(self):
#         def loss(point, data):
#             """
#             RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
#             """
#             size = len(data)
#             beta, gamma = point
#             def SIR(t, y):
#                 S = y[0]
#                 I = y[1]
#                 R = y[2]
#                 return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
#             solution = solve_ivp(SIR, [0, size], self.get_init_point(),
#                                  t_eval=np.arange(0, size, 1), vectorized=True)
#             return np.sqrt(np.mean((solution.y[1] - data)**2))

#         return loss

#     def train(self):
#         """
#         Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
#         """

#         print('Training...')
#         optimal = minimize(
#             self.loss,
#             [0.001, 0.001],
#             args=(self.data),
#             method='L-BFGS-B',
#             bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
#         )
#         beta, gamma = optimal.x
#         self._beta = beta
#         self._gamma = gamma


#     def predict(self):
#         print('Predict...')
#         beta, gamma = self._beta, self._gamma

#         new_index, extended_actual, prediction = self._predict(beta, gamma, self.data)
#         df = pd.DataFrame({
#             'Actual': extended_actual,
#             'S': prediction.y[0],
#             'I': prediction.y[1],
#             'R': prediction.y[2]
#         }, index=new_index)
#         fig, ax = plt.subplots(figsize=(15, 10))
#         ax.set_title(f"{self.country} beta={beta}, gamma={gamma}")
#         df.plot(ax=ax)
#         fig.savefig(f"{self.country}.png")



# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description = "COVID-19 estimation of params")
#     parser.add_argument('dataset_path', type=str, help='dataset path')
#     parser.add_argument('country', type=str, help='country')
#     parser.add_argument('--population', type=int, default=30000, help='Population [default: 100k]')
#     args = parser.parse_args()    

#     model = Learner(args.country, args.dataset_path, args.population)
#     model.train()
#     model.predict()
