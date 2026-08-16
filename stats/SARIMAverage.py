from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf

class SARIMAverage:
    def __init__(self):
        pass

    def buildModel(self, train, test, toDate):
        model = SARIMAX(
            train,
            order = (3, 1, 1),
            seasonal_order = (0, 0, 0, 0)
        )
        results = model.fit(disp = 0)
        forecast = results.forecast()
        predicted = forecast[0]
        print(f'[SARIMA] Predicted Value On {toDate}: {round(predicted, 2)}')
        print(f'[SARIMA] Actual Value On {toDate}: {round(test[0], 2)}')

if __name__ == "__main__":
    sarima = SARIMAverage()
    fromDate = '2026-06-10'
    toDate = '2026-08-15'
    quotes = yf.download(
        'FB',
        start = fromDate,
        end = toDate
    )
    closes = quotes['Close'].values
    train, test = closes[:-1], closes[-1]
    sarima.buildModel(train, test, toDate)