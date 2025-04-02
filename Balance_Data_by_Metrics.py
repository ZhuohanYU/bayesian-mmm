
# balance_data_by_metric
import pandas as pd

def balance_data_by_metric(df, metric_col, max_per_group=None, random_state=42):
    """
    Balances a DataFrame by limiting the number of rows per unique value in metric_col.
    
    Parameters:
    - df (pd.DataFrame): Original dataset
    - metric_col (str): Column to balance on (e.g., 'region', 'high_sales_bin')
    - max_per_group (int): Max rows per group (defaults to size of smallest group)
    
    Returns:
    - pd.DataFrame: Balanced dataset
    """
    grouped = df.groupby(metric_col)
    
    if max_per_group is None:
        max_per_group = grouped.size().min()
    
    balanced_df = grouped.apply(lambda x: x.sample(n=max_per_group, random_state=random_state))
    balanced_df = balanced_df.reset_index(drop=True)
    
    return balanced_df
