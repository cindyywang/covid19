import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

# Definition of the logistic function
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

def main():
  new_cases_sh = [2, 8, 16, 19, 28, 48, 55, 65, 80, 75, 83, 65, 169, 139, 202, 157, 260, 374, 503, 758, 896, 981, 983, 1609, 2264, 2676, 3500, (96 - 21 + 4381), (326 - 18 + 5656), (5298 + 355 - 16), (358 + 4144 -20), (260 + 6051 - 2), (7788 + 438 - 73), (8581 + 425 - 71), (13086 + 268 - 4), (16766 + 311 - 40), (19660 + 322 -15), (20398 + 824 - 323), (22609 + 1015 - 420), (23937 + 1006 - 191), (25173 + 914 - 47), (22348 + 994 - 273), (25141 + 1189 - 23), (25146 + 2573 - 114), (19872 + 3200 - 307), (19923 + 3590 - 922), (21582 + 3238 - 1177), (19831 + 2417 - 853 ), 19442, (2494 + 16407 - 533), (2634 + 15861 - 459), (15698 + 1931 - 143), (20634 + 2736 - 1120), (19657 + 1401 - 541), (16983 + 2472 - 846), (15319 + 1661 - 968), (11956 + 1606 - 1253), (9330 + 1292 - 858), (9545 + 5487 - 5062), (1249 + 8932 - 985), (7084 + 788 - 683), (6606 + 727 - 529), (5395 + 274 - 155), (4722 + 260 - 151), (4390 + 261 - 185), (4024 + 245 - 181), (3961 + 253 -175), (3760 + 215 - 135), (3625 + 322 - 230), (2780 + 234 - 156), (228 + 1259 - 198), (1305 + 144 - 106), (1869 + 227 - 167), (1487 + 194 - 140), (1203 + 166 - 111), (869 + 69 - 42), (746 + 77 - 46), (759 + 96 - 56), (637 + 82 - 48), (770 + 88 - 71), (784 + 84 - 49), (570 + 52 - 29), (503 + 55 - 30), (422 + 58 - 39), (343 + 44 - 32), (290 + 48 - 31), (219 + 45 - 33), (131 + 39 - 18), (29 + 93 - 18), (6 + 61 - 2)]
  #for day, cases in enumerate(new_cases_sh):
    #print(day, " daily new cases in sh: ", cases)
  plt.plot(new_cases_sh)
  plt.savefig('covid_20220529.png')
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
  plt.savefig('prediction_20220529.png')
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
  plt.savefig('new_beta_prediction_20220529.png')
  plt.show()

  # #Calculate mean and Standard deviation.
  # mean = np.mean(y_data)
  # sd = np.std(y_data)

  # #Apply function to the data.
  # pdf = normal_dist(y_data,mean,sd)

  # #Plotting the Results
  # plt.plot(y_data,pdf , color = 'red')
  # plt.xlabel('Data points')
  # plt.ylabel('Probability Density')
  # plt.savefig('normal_distribution_20220523.png')
  # plt.show()

  mu = np.mean(x_data)
  sigma = np.std(x_data) #You manually calculated it but you can also use this built-in function
  data = np.random.normal(mu, sigma, len(new_cases_sh))
  count, bins, ignored = plt.hist(data, 30, density=True, stacked=True)
  plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
  plt.savefig('normal_distribution_20220529.png')
  plt.show()

if __name__ == '__main__':
  main()
