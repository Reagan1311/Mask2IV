import pandas as pd

# Load the CSV file
df = pd.read_csv('/workspace/exp_outputs/data_hoi4d.csv')

# List to store indices of rows to be deleted
rows_to_delete = []

# Iterate through each row and check the start and end frames
for index, row in df.iterrows():
    start_frame = row['start_frame']
    end_frame = row['end_frame']
    frame_length = end_frame - start_frame + 1
    
    if frame_length < 12:
        rows_to_delete.append(index)
    elif frame_length < 16:
        new_end_frame = end_frame + (16 - frame_length)
        if new_end_frame > 299:
            new_start_frame = max(0, start_frame - (16 - frame_length))
            df.at[index, 'start_frame'] = start_frame
        else:
            df.at[index, 'end_frame'] = new_end_frame

# Delete rows with frame length smaller than 12
df.drop(rows_to_delete, inplace=True)

# Save the modified DataFrame back to a CSV file
df.to_csv('/workspace/exp_outputs/hoi4d_data_filtered.csv', index=False)