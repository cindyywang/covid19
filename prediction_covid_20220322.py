import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Definition of the logistic function
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

def main():
  new_cases_sh = [2, 8, 16, 19, 28, 48, 55, 65, 80, 75, 83, 65, 169, 139, 202, 157, 260, 374, 503, 758, 896, 981]
  plt.plot(new_cases_sh)
  plt.savefig('covid_20220322.png')
  plt.show()
  for i in range(len(new_cases_sh) - 1):
    growth_factor = float(new_cases_sh[i + 1]) / float(new_cases_sh[i])
    th = ""
    if( i + 1 == 1):
      th = "st"
    elif( i + 1 == 2):
      th = "nd"
    else:
      th = "th"
    print(i + 1, th + " growth factor: ", growth_factor)

  x_data, y_data = np.array(list(range(1, len(new_cases_sh) + 1))), np.array(new_cases_sh)

  # Choosing initial arbitrary beta parameters
  beta_1 = 0.09
  beta_2 = 305

  # application of the logistic function using beta
  Y_pred = sigmoid(x_data, beta_1, beta_2)

  # point prediction
  plt.plot(x_data, Y_pred * 15000000000000., label = "Model")
  plt.plot(x_data, y_data, 'ro', label = "Data")
  plt.title('Data Vs Model')
  plt.legend(loc ='best')
  plt.ylabel('Cases')
  plt.xlabel('Day Number')
  plt.savefig('prediction_20220322.png')
  plt.show()

  xdata = x_data / max(x_data)
  ydata = y_data / max(y_data)

  popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=5000)
  # print the final params
  print("beta_1 = % f, beta_2 = % f" % (popt[0], popt[1]))

  x = np.linspace(0, 40, 4)
  x = x / max(x)

  plt.figure(figsize = (8, 5))

  y = sigmoid(x, *popt)

  plt.plot(xdata, ydata, 'ro', label ='data')
  plt.plot(x, y, linewidth = 3.0, label ='fit')
  plt.title("Data Vs Fit model")
  plt.legend(loc ='best')
  plt.ylabel('Cases')
  plt.xlabel('Day Number')
  plt.savefig('new_beta_prediction_20220322.png')
  plt.show()

if __name__ == '__main__':
  main()
