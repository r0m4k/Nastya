import pandas as pd
import json
import random

# Load the four datasets
df1 = pd.read_csv('datasetCreation/data_training_1.csv')
df2 = pd.read_csv('datasetCreation/data_training_2.csv')
df3 = pd.read_csv('datasetCreation/data_training_3.csv')
df4 = pd.read_csv('datasetCreation/data_training.csv')

# Concatenate all dataframes
full_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Rename columns for consistency
full_df = full_df.rename(columns={'aspect_term': 'aspectTerm'})

# Drop rows with missing values
full_df.dropna(subset=['aspectTerm', 'sentence', 'sentiment', 'from', 'to'], inplace=True)


# Open the output file
with open('llama_finetuning_instruct_data.jsonl', 'w') as f:
    # Iterate over each row to create the JSON objects
    for index, row in full_df.iterrows():
        # Create the prompt and completion texts

        # FOR FINETUNING
        # prompt_text = f"What is the sentiment (positive, negative, neutral) for the aspect term {row['aspectTerm']} (from {row['from']} to {row['to']} character) in the sentence {row['sentence']}?"

        # FOR INSTRUCTION FINETUNING
        prompt_variations = [
        f"Classify the sentiment (positive, negative, neutral) expressed toward the aspect term {row['aspectTerm']} located at characters {row['from']}-{row['to']} in the sentence: {row['sentence']}",
        f"For the sentence '{row['sentence']}', what is the sentiment orientation (positive / negative / neutral) of the aspect '{row['aspectTerm']}' spanning characters {row['from']} to {row['to']}?",
        f"Identify whether the aspect term '{row['aspectTerm']}' (positions {row['from']}-{row['to']}) carries a positive, negative, or neutral sentiment in the following sentence: {row['sentence']}",
        f"Return only the sentiment label (positive, negative, neutral) for the aspect '{row['aspectTerm']}' appearing between character {row['from']} and {row['to']} in: {row['sentence']}",
        f"Determine the sentiment category (positive / negative / neutral) toward '{row['aspectTerm']}' that starts at {row['from']} and ends at {row['to']} in this sentence: {row['sentence']}",
        f"Assess whether the sentiment directed at the aspect term '{row['aspectTerm']}' (chars {row['from']}-{row['to']}) is positive, negative, or neutral in the sentence: {row['sentence']}",
        f"Label the emotional polarity (positive, negative, neutral) of '{row['aspectTerm']}' found from character {row['from']} to {row['to']} within the sentence '{row['sentence']}'.",
        f"Given '{row['sentence']}', indicate if the sentiment toward the aspect term '{row['aspectTerm']}' (index range {row['from']}-{row['to']}) is positive, negative, or neutral.",
        f"Classify the sentiment polarity (positive / negative / neutral) associated with the aspect '{row['aspectTerm']}' positioned at {row['from']}-{row['to']} in the sentence: {row['sentence']}",
        f"In the following sentence, is the sentiment toward '{row['aspectTerm']}' (characters {row['from']}-{row['to']}) positive, negative, or neutral? {row['sentence']}"
        ]

        prompt_text = random.choice(prompt_variations)
        completion_text = row['sentiment']

        # Create the dictionary
        json_record = {
            "prompt": prompt_text,
            "completion": completion_text
        }

        # Convert dictionary to JSON string and write to file
        f.write(json.dumps(json_record) + "\n")