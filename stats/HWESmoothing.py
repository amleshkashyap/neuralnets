from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf

class HWESmoothing:
    def __init__(self):
        pass

    def buildModel(self, train, test, toDate):
        model = ExponentialSmoothing(train)
        results = model.fit()
        predicted = results.forecast()[0]
        print(f'[HWES] Predicted Value On {toDate}: {round(predicted, 2)}')
        print(f'[HWES] Actual Value On {toDate}: {round(test[0], 2)}')

if __name__ == '__main__':
    hwes = HWESmoothing()
    fromDate = '2026-06-10'
    toDate = '2026-08-15'
    quotes = yf.download(
        'FB',
        start = fromDate,
        end = toDate
    )
    closes = quotes['Close'].values
    train, test = closes[:-1], closes[-1]
    hwes.buildModel(train, test, toDate)