from src.models.BagLearner import BagLearner
from src.models.RTLearner import RandomTreeLearner

from src.features.indicators import bollinger_bands_percentage, relative_strength_index, rate_of_change, moving_average_convergence_divergence, stochastic_oscillator

import datetime as dt

learner = BagLearner(
    learner=RandomTreeLearner,
    kwargs={"leaf_size": 5},
    bags=10,
    mode="gini",
)

def add_evidence(  		  	   		 	 	 			  		 			 	 	 		 		 	
    self,  		  	   		 	 	 			  		 			 	 	 		 		 	
    symbol="JPM",  		  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    sv=10000,  		  
    future_window=5,
    threshold=0.005,	   		  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	  	  
    backfilled_prices = ut.get_data([symbol], pd.date_range(sd - dt.timedelta(days=200), ed))[symbol]
    daily_prices = ut.get_data([symbol], pd.date_range(sd, ed))[symbol]
    backfill_len = len(backfilled_prices) - len(daily_prices)

    bbp_signals = bollinger_bands_percentage(backfilled_prices)[backfill_len:]
    rsi_signals = relative_strength_index(backfilled_prices)[backfill_len:]
    roc_signals = rate_of_change(backfilled_prices)[backfill_len:]
    macd_signals = moving_average_convergence_divergence(backfilled_prices)[backfill_len:]
    so_signals = stochastic_oscillator(backfilled_prices)[backfill_len:]
    
    x = np.column_stack((bbp_signals, rsi_signals, roc_signals, macd_signals, so_signals))
    y = np.zeros(len(daily_prices))

    net_return = ((daily_prices.shift(-future_window).values * (1 - self.impact))/ (daily_prices.values  * (1 + self.impact)) - 1) \
    - ((2.0 * self.commission) / (1000 * daily_prices.values * (1 + self.impact)))

    y[net_return >  threshold] =  1
    y[net_return < -threshold] = -1

    self.learner.add_evidence(x, y)		
