import matplotlib
import matplotlib.pyplot as plt

def main():
  new_cases_sh = [2, 8, 16, 19, 28, 48, 55, 65, 80, 75, 83]
  plt.plot(new_cases_sh)
  plt.savefig('covid_20220310.png')
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

if __name__ == '__main__':
  main()
