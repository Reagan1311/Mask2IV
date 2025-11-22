import pandas as pd
import csv
import os

df_train = "/workspace/exp_outputs/train_bdv2.csv"
df_val = "/workspace/exp_outputs/val_bdv2.csv"
df_test = "/workspace/exp_outputs/test_bdv2.csv"

df = pd.read_csv('/workspace/exp_outputs/bdv2_data.csv')
df = df[df['path'].apply(lambda p: os.path.exists(os.path.join(p, 'masks')))]  # 13781
df = df[df['path'].apply(lambda p: len(os.listdir(os.path.join(p, 'masks'))) > 0 )]

test_df = df.sample(n=200, random_state=42)  # 250 rows, uniformly sampled
df.drop(test_df.index, inplace=True)

val_df = df.sample(n=50, random_state=42)  # 100 rows, uniformly sampled
df.drop(val_df.index, inplace=True)

df.to_csv(df_train, index=False)
test_df.to_csv(df_test, index=False)
val_df.to_csv(df_val, index=False)


