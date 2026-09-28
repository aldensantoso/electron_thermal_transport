from src.run_model import get_clean_df

df = get_clean_df()
print(df.shape)
print(df.columns[:20])
print(df.head())