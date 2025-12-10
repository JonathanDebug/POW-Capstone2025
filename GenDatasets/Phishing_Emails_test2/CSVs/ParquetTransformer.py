import pandas as pd
import os

print(os.getcwd())
cur_path= os.path.dirname(__file__)

#Define output path 
INPUT_PATH= 'GenDatasets/Phishing_Emails_test2/dataset'
OUTPUT_PATH = os.path.join(cur_path, 'output') 


print(INPUT_PATH)
#Load the parquet file. 

# count = 0
df_list = []


# for file in os.listdir(INPUT_PATH):
#   file_path = os.path.join(INPUT_PATH, file)
#   df = pd.read_parquet(file_path)
#   print(file + "has been read, converting to csv")
  
#   output_file = os.path.join(OUTPUT_PATH, f"synthCartero{count}.csv")
#   df.to_csv(output_file, index=False)
#   count+=1
#   print(file + "has been converted, following the other one.")


for csv_file in os.listdir('GenDatasets/Phishing_Emails_test2/CSVs/BothCSV_output'):
  file_path = os.path.join('GenDatasets/Phishing_Emails_test2/CSVs/BothCSV_output/', csv_file)
  df = pd.read_csv(file_path)
  df_list.append(df)



final_df = pd.concat(df_list, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42)

final_output_path = os.path.join(cur_path, "Synth_POW.csv")
final_df.to_csv(final_output_path)





  
