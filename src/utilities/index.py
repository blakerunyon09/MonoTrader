import pandas as pd
import os

def symbol_to_path(symbol, base_dir=None):  		  	   		 	 	 			  		 			 	 	 		 		 		  	   		 	 	 			  		 			 	 	 		 		 	
    if base_dir is None:  		  	   		 	 	 			  		 			 	 	 		 		 	
        base_dir = os.environ.get("MARKET_DATA_DIR", "data")  		  	   		 	 	 			  		 			 	 	 		 		 	
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))  		  	   		 	 	 			  		 			 	 	 		 		 	

def get_data(symbol, dates, colname="Close"):  		  	   		 	 	 			  		 			 	 	 		 		 	 		  	   		 	 	 			  		 			 	 	 		 		 	
    df = pd.DataFrame(index=dates)  	

    df_temp = pd.read_csv(
        symbol_to_path(symbol),
        index_col=0,
        parse_dates=True,
        na_values=["nan"],
    )
    df_temp.index.name = "Date"

    df_temp = df_temp.rename(columns={colname: symbol})
    df = df.join(df_temp)

    return df