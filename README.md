# MontePy
*Monte Carlo simulation using Numpy and Pandas*

![alt text](./dice.jpeg)

## References
- [Monte Carlo Simulations with Python](https://towardsdatascience.com/monte-carlo-simulations-with-python-part-1-f5627b7d60b0)
- [Numpy Random Distribution](https://docs.scipy.org/doc/numpy/reference/routines.random.html)
- [Python Monte Carlo](https://pbpython.com/monte-carlo.html)
- [Numpy Binomial Distribution](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html#numpy.random.binomial)
- [Numpy Normal Distribution](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html#numpy.random.normal)
- [Numpy Exponential Distribution](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential)
- [Pandas Eval](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html)

## Installation
```
git clone ssh://git@github.com/MartinCastroAlvarez/monte-py
cd monte-py
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

## Usage
Create a file called `dataset.json` and put this config inside:
```
{
    "features": {
        "Sales": {
            "distribution": "normal",
            "avg": 100,
            "std": 0.8,
            "min": 0
        },
        "Price": {
            "distribution": "normal",
            "avg": 10,
            "std": 0.2,
            "min": 0
        },
        "FixedCost": {
            "distribution": "normal",
            "avg": 100,
            "std": 0.5,
            "min": 0
        },
        "VariableCost": {
            "distribution": "normal",
            "avg": 8,
            "std": 2,
            "min": 0
        }
    },
    "targets": [{
        "Revenue": "Price * Sales",
        "Cost": "FixedCost + VariableCost * Sales"
    }, {
        "Profit": "Revenue - Cost"
    }]
}
```
Execute the following command to run the simulation:
```
python3 simulate.py  --simulations 100 --path dataset.json
```
Look at the output:
```
            Sales       Price   FixedCost  VariableCost      Revenue         Cost      Profit
count  100.000000  100.000000  100.000000    100.000000   100.000000   100.000000  100.000000
mean   100.016270    9.985491   99.940654      7.858189   998.716079   886.028664  112.687415
std      0.823051    0.197306    0.485513      1.887991    21.583492   189.225182  189.462751
min     97.984897    9.473178   98.542858      3.344354   939.624517   434.246999 -438.177117
25%     99.490436    9.845898   99.620062      6.867120   983.393390   787.918456  -12.257061
50%    100.028057    9.989368   99.908596      8.057500   999.504628   906.059768   98.668392
75%    100.542664   10.142087  100.307828      9.117511  1014.255521  1010.227241  225.900207
max    102.427096   10.404904  100.937590     13.529123  1044.997147  1435.870633  526.558347
```
In this example, the expected profit is `$112.687415` after simulating the scenario 100 times.
