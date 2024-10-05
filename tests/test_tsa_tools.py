from tsatools import tsa_tools
import pandas as pd

def test_split_ts():
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-01-20')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D') 
    data = {
        'date':date_range,
        'x':list(range(0,20,1))
    }
    df = pd.DataFrame(data)
    x, y = tsa_tools.split_ts(df, 0.8)
    assert len(x) == 16 and len(y) == 4
