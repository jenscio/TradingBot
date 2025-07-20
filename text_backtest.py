import yfinance as yf
import numpy as npexit
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as pltpip

nvd = yf.download('NVDA')
nvd.head()