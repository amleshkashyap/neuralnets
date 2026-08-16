from statsmodels.tsa.ar_model import AutoReg
import yfinance as yf

class AutoRegressive:
    def __init__(self):
        pass

    def buildModel(self, quotes, lags = 2):
        model = AutoReg(
            quotes['Close'],
            lags = lags
        )
        modelFit = model.fit()
        print(modelFit.summary())
        print(modelFit.params)


if __name__ == "__main__":
    ar = AutoRegressive()
    fromDate = '2026-06-10'
    toDate = '2026-08-15'
    quotes = yf.download(
        'FB',
        start = fromDate,
        end = toDate
    )
    ar.buildModel(quotes, 2)