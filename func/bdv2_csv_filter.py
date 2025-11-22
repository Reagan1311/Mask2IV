import numpy as np
import pandas as pd

# df = pd.read_csv('/workspace/exp_outputs/bdv2_data.csv')
df = pd.read_csv('../train_bdv2.csv')

df_filter = df[(df['duration'] <= 50) & (df['confidence']>=0.4)] # 0.2: 16604; 0.4: 16030

df_action_filter = df_filter[df_filter['caption'].str.contains(r'put|push|open|close|take|fold|unfold|move', case=False, na=False)]
df_action_filter_ = df_action_filter[~df_action_filter['caption'].str.contains(r'rectangular|hexagon|cube|block|arch|rectangle|cylinder|arc|cub|triangle|ark', case=False, na=False)]

# df_action_filter_['caption'].to_csv('/workspace/exp_outputs/bdv2_data_filter_caption.csv', index=False)
df_action_filter_['caption'].to_csv('../bdv2_data_filter_caption.csv', index=False)
import pdb; pdb.set_trace()

