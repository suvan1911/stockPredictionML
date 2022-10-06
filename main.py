from tabnanny import verbose
import pandas as pd
import numpy as np
import tensorflow


model = tensorflow.keras.models.load_model('model')

def make_trades(df, holdings, current_money, day):
    to_spend = current_money
    prices = df.iloc[:, -50:] 
    preds = model.predict([prices], verbose=0).flatten()
    current_prices = df.iloc[:, -1]
    ret = {}

    p_diff = pd.Series((preds - current_prices)/current_prices * 100, dtype=float).sort_values(ascending=False)
    gain_sum = sum([i for i in p_diff if i>0])

    top = 10
    topind = [i for i,v in p_diff.items() if v > 0][:top]
    for i,v in holdings.items():
        if v>0 and i not in topind:
            ret[i] = -v
    for i,v in p_diff.items():
        if v>0 and i in topind:
            ret[i] = ((v/gain_sum)*to_spend) // current_prices[i]
    
    return ret


