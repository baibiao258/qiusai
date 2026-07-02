import pandas as pd
from run_backtest_with_csv import eval_backtest

print("Eval Predictions Log with Decomp")
df = pd.read_csv('/root/data/predictions_log.csv')
eval_backtest(df)
