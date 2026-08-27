from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

class ARIMAverage:
    def __init__(self):
        pass

    def buildModel(self, quotes, toDate):
        closes = quotes['Close'].values
        train, test = closes[:-1], closes[-1]
        model = ARIMA(
            train,
            order = (5, 2, 3)
        )
        results = model.fit()
        forecast = results.forecast()
        predicted = forecast[0]
        print(f'[ARIMA] Predicted Value On {toDate}: {round(predicted, 2)}')
        print(f'[ARIMA] Actual Value On {toDate}: {round(test[0], 2)}')

if __name__ == '__main__':
    arima = ARIMAverage()
    fromDate = '2026-06-10'
    toDate = '2026-08-15'
    quotes = yf.download(
        'FB',
        start = fromDate,
        end = toDate
    )
    arima.buildModel(quotes, toDate)