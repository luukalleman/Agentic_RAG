import pandas as pd


# Sample DataFrame with order data
data = {
    'order_number': ['12345', '23456', '34567'],
    'status': ['Shipped', 'Processing', 'Delivered'],
    'estimated_delivery': ['2024-11-01', '2024-11-05', '2024-10-30']
}
df_orders = pd.DataFrame(data)