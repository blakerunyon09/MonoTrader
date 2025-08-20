from src.models.BagLearner import BagLearner
from src.models.RTLearner import RandomTreeLearner

from src.features.indicators import bollinger_bands_percentage, relative_strength_index, rate_of_change, moving_average_convergence_divergence, stochastic_oscillator

import datetime as dt
import src.utilities.index as ut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

learner = BagLearner(
    learner=RandomTreeLearner,
    kwargs={"leaf_size": 5},
    bags=10,
    mode="gini",
)

def add_evidence(  		  	   		 	 	 			  		 			 	 	 		 		 			  	   		 	 	 			  		 			 	 	 		 		 	
    symbol="AAPL",  	  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2023, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2024, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    sv=10000,  		  
    future_window=5,
    threshold=0.005,	   		  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	  	  
    backfilled_prices = ut.get_data(symbol, pd.date_range(sd - dt.timedelta(days=200), ed))[symbol]
    daily_prices = ut.get_data(symbol, pd.date_range(sd, ed))[symbol]
    backfill_len = len(backfilled_prices) - len(daily_prices)

    bbp_signals = bollinger_bands_percentage(backfilled_prices)[backfill_len:]
    rsi_signals = relative_strength_index(backfilled_prices)[backfill_len:]
    roc_signals = rate_of_change(backfilled_prices)[backfill_len:]
    macd_signals = moving_average_convergence_divergence(backfilled_prices)[backfill_len:]
    so_signals = stochastic_oscillator(backfilled_prices)[backfill_len:]
    
    x = np.column_stack((bbp_signals, rsi_signals, roc_signals, macd_signals, so_signals))
    y = np.zeros(len(daily_prices))

    net_return = ((daily_prices.shift(-future_window).values * (1))/ (daily_prices.values  * (1)) - 1) \
    - ((2) / (1000 * daily_prices.values * (1)))

    y[net_return >  threshold] =  1
    y[net_return < -threshold] = -1

    learner.add_evidence(x, y)		

def testPolicy(  		  	   		 	 	 			  		 			 	 	 		 		 	  		  	   		 	 	 			  		 			 	 	 		 		 	
    symbol="AAPL",  		  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2023, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2024, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    sv=100000,  	
    return_order_book=False	  	   		 	 	 			  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 		  	   		 	 	 			  		 			 	 	 		 		 	
    backfilled_prices = ut.get_data(symbol, pd.date_range(sd - dt.timedelta(days=200), ed))[symbol]
    daily_prices = ut.get_data(symbol, pd.date_range(sd, ed))[symbol]
    backfill_len = len(backfilled_prices) - len(daily_prices)

    bbp_signals = bollinger_bands_percentage(backfilled_prices)[backfill_len:]
    rsi_signals = relative_strength_index(backfilled_prices)[backfill_len:]
    roc_signals = rate_of_change(backfilled_prices)[backfill_len:]
    macd_signals = moving_average_convergence_divergence(backfilled_prices)[backfill_len:]
    so_signals = stochastic_oscillator(backfilled_prices)[backfill_len:]

    x = np.column_stack((bbp_signals, rsi_signals, roc_signals, macd_signals, so_signals))
    trades = learner.query(x)
    trades = pd.Series(trades, index=daily_prices.index)

    order_book = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    daily_prices = ut.get_data(symbol, pd.date_range(sd, ed))
    final_trades = pd.DataFrame(0, index=daily_prices.index, columns=[symbol])

    position = 0

    for today in daily_prices.index[:-1]:
        if trades[today] > 0 and position < 1000:
            if not order_book.empty and order_book.iloc[-1]['Order'] == 'BUY':
                continue

            new_order = pd.DataFrame({
                'Date': today,
                'Symbol': symbol,
                'Order': 'BUY',
                'Shares': 1000 if order_book.empty else 2000
            }, index=[order_book.shape[0]])
            
            final_trades.loc[today, symbol] = 1000 if order_book.empty else 2000
            order_book = new_order if order_book.empty else pd.concat([order_book, new_order])
            position += 1000 if order_book.empty else 2000

        elif trades[today] < 0 and position > -1000:
            if not order_book.empty and order_book.iloc[-1]['Order'] == 'SELL':
                continue

            new_order = pd.DataFrame({
                'Date': today,
                'Symbol': symbol,
                'Order': 'SELL',
                'Shares': 1000 if order_book.empty else 2000
            }, index=[order_book.shape[0]])

            final_trades.loc[today, symbol] = -1000 if order_book.empty else -2000
            order_book = new_order if order_book.empty else pd.concat([order_book, new_order])
            position -= 1000 if order_book.empty else 2000	  	   		 	 	 			  		 			 	 	 		 		 	

    # # Handle first day if no trades were made in order_book
    if order_book.empty or (order_book.iloc[0]['Date'] != daily_prices.index[0]):
        new_order = pd.DataFrame({
            'Date': daily_prices.index[0],
            'Symbol': symbol,
            'Order': 'OUT',
            'Shares': 0
        }, index=[0])
        order_book = pd.concat([new_order, order_book])

    # Handle last day
    if position != 0 and not order_book.empty:
        new_order = pd.DataFrame({
            'Date': daily_prices.index[-1],
            'Symbol': symbol,
            'Order': 'SELL' if position > 0 else 'BUY',
            'Shares': 1000
        }, index=[order_book.shape[0]])
        final_trades.loc[daily_prices.index[-1], symbol] += -1000 if position > 0 else 1000
        order_book = pd.concat([order_book, new_order])

    if return_order_book:
        return final_trades, order_book

    return final_trades    

def compute_portvals(order_book, start_val=100000, commission=9.95, impact=0.005):  
    order_book = order_book.set_index("Date")
    order_book.sort_index(inplace=True)
    order_book.index = pd.to_datetime(order_book.index).normalize()

    symbols = order_book["Symbol"].unique()

    daily_prices = ut.get_data(symbols[0], pd.date_range(order_book.index.min(), order_book.index.max()))

    trades = pd.DataFrame(0, index=daily_prices.index, columns=symbols)
    cash = pd.DataFrame(0, index=daily_prices.index, columns=["Balance"])
    cash.loc[cash.index[0], 'Balance'] = start_val

    for date in daily_prices.index:
        i = cash.index.get_loc(date)

        if i > 0:
            cash.loc[date, 'Balance'] = cash.loc[cash['Balance'].index[i - 1], 'Balance']
            trades.loc[date] = trades.loc[trades.index[i - 1]]
            
        if date in order_book.index:
            daily_orders = order_book.loc[date]

            if isinstance(daily_orders, pd.Series):
                daily_orders = pd.DataFrame([daily_orders])

            for _, order in daily_orders.iterrows():
                isBuy = order['Order'] == 'BUY'

                net = order['Shares'] * daily_prices.loc[date, order['Symbol']]

                net *= (1 + impact) if isBuy else (1 - impact)
                net = net + commission if isBuy else net - commission

                trades.loc[date, order['Symbol']] += order['Shares'] if isBuy else -1 * order['Shares']

                cash.loc[date, 'Balance'] = cash.loc[date, 'Balance'] - net if isBuy else cash.loc[date, 'Balance'] + net

    return (trades * daily_prices).sum(axis=1) + cash["Balance"]

def get_cumulative_return(orders):
    portfolio_value = compute_portvals(orders)
    portfolio_value = portfolio_value / portfolio_value.iloc[0]
    return portfolio_value, (portfolio_value.iloc[-1] - portfolio_value.iloc[0]) / portfolio_value.iloc[0]
      
def run():
    symbol = "AAPL"
    # bags = [1, 5, 10, 20, 50, 500]
    # leaf_size = [5, 10, 20]
    # future_window = [1, 2, 3]
    # threshold = [0.0,1e-4,5e-4,7.5e-4,1e-3,2e-3,3e-3,5e-3,7.5e-3,1e-2,1.5e-2,2e-2,3e-2,5e-2,1e-1]

    add_evidence(symbol=symbol, sd=dt.datetime(2023, 1, 1), ed=dt.datetime(2024, 12, 31), future_window=1, threshold=0.05)

    _, t_in_order_book = testPolicy(symbol=symbol, sd=dt.datetime(2023, 1, 1), ed=dt.datetime(2024, 12, 31), return_order_book=True)
    _, t_out_order_book = testPolicy(symbol=symbol, sd=dt.datetime(2025, 1, 1), ed=dt.datetime(2025, 8, 1), return_order_book=True)

    print(t_in_order_book)

    in_gini_portvalue, in_gini_cumulative_return = get_cumulative_return(t_in_order_book)
    out_gini_portvalue, out_gini_cumulative_return = get_cumulative_return(t_out_order_book)

    # plt.figure()
    # plt.title(f"In Sample Portfolio Value {symbol}")

    # plt.xlabel("Date")
    # plt.ylabel("Normalized Portfolio Value")

    # plt.plot(in_gini_portvalue.index, in_gini_portvalue, label="Gini Learner", color='blue')

    # plt.legend()
    # plt.grid(True)
    # plt.savefig("assets/In_Sample_Portfolio.png")
    # plt.close()

    # plt.figure()
    # plt.title(f"Out of Sample Portfolio Value {symbol}")

    # plt.xlabel("Date")
    # plt.ylabel("Normalized Portfolio Value")

    # plt.plot(out_gini_portvalue.index, out_gini_portvalue, label="Gini Learner", color='blue')

    # plt.legend()
    # plt.grid(True)
    # plt.savefig("assets/Out_of_Sample_Portfolio.png")
    # plt.close()

    # print(
    #     f"In Sample Gini Cumulative Return: {in_gini_cumulative_return:.6f}, \n"
    #     f"Out of Sample Gini Cumulative Return: {out_gini_cumulative_return:.6f}, \n"
    # )

run()