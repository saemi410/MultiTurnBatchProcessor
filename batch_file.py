import json
import os
import csv
import time
from datetime import datetime
from openai import OpenAI

# * See https://platform.openai.com/docs/guides/batch?lang=python for more details

class MultiTurnBatchProcessor:
    def __init__(self, model, max_tokens, initial_messages, custom_id_list=None):
        self.model = model # * model name (eg. gpt-3.5-turbo-0125)
        self.max_tokens = max_tokens # * max tokens for each request
        self.messages_list = initial_messages # * conversation history for each request
        
        if custom_id_list is None:  # * identifier for each request
            self.custom_id_list = [f"request-{i}" for i in range(len(self.messages_list))]
        else:
            self.custom_id_list = custom_id_list
        self.output_dir = self.create_output_directory()
        self.turn_count = 0
        
        # ! NOTE: please set the OPENAI_API_KEY in the environment variable
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # * find the index of the custom_id and add the message to the messages_list
    def add_messages(self, message, custom_id):
        index = self.custom_id_list.index(custom_id)
        self.messages_list[index].append(message)
    
    def batch_element(self, messages, custom_id):
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens
            }
        }
    
    # * create a batch for each turn
    def create_batch(self):            
        with open(os.path.join(self.output_dir, f"turn{self.turn_count}.jsonl"), "w") as f:
            for message, custom_id in zip(self.messages_list, self.custom_id_list):
                f.write(json.dumps(self.batch_element(message, custom_id)) + "\n")
        
        print(f"Creating batch for turn {self.turn_count}...")
        self.batch_input_file = self.client.files.create(
            file=open(os.path.join(self.output_dir, f"turn{self.turn_count}.jsonl"), "rb"),
            purpose="batch"
        )
        
        while not self.is_uploaded():
            print(f"Waiting for batch file to be uploaded...")
            time.sleep(5)
        
        # * Once you've successfully uploaded your input file, you can use the input File object's ID to create a batch
        self.batch_file_id = self.batch_input_file.id
        metadata = self.client.batches.create(
            input_file_id=self.batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "turn " + str(self.turn_count)
            }
        )
        self.batch_id = metadata.id
        
        print(f"---------Metadata for turn {self.turn_count}----------")
        print(metadata)
        print("-------------------------------------------")
                
        self.turn_count += 1
    
    def is_completed(self):
        batch = self.client.batches.retrieve(self.batch_id)
        batch_status = batch.status
        print(f"Current batch status of {self.batch_id}: {batch_status}")
        return batch_status == "completed"
    
    def is_uploaded(self):
        return self.batch_input_file.id is not None
    
    def get_batch_result(self):
        while not self.is_completed():
            time.sleep(10)
        batch = self.client.batches.retrieve(self.batch_id)
        if batch.error_file_id is not None:
            print(f"There is an error in the batch! {self.client.files.content(batch.error_file_id).text}")
            return None
        elif batch.output_file_id is not None:
            self.output_file_id = batch.output_file_id
            batch_result = self.client.files.content(self.output_file_id).text # * jsonl style
            batch_result = [json.loads(line) for line in batch_result.split("\n") if line]
            
            return batch_result
        else:
            print("There is no output file in the batch!")
            return None
        
    def execute_one_turn(self):
        self.create_batch()
        batch_result = self.get_batch_result()
        for item in batch_result:
            custom_id = item["custom_id"]
            new_message = item["response"]["body"]["choices"][0]["message"]
            self.add_messages(new_message, custom_id)

    def create_output_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("logs", self.model, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
    
    def save_messages(self):
        with open(os.path.join(self.output_dir, "messages_list.json"), "w") as f:
            json.dump(self.messages_list, f)
    
def load_harmbench_dataset():
    initial_messages_list = []
    custom_id_list = []
    with open("harmbench_behaviors_text_test.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == "standard":
                initial_messages_list.append([{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": row[0]}])
                custom_id_list.append(row[5])
    return initial_messages_list, custom_id_list
         
if __name__ == "__main__":
    model = "gpt-3.5-turbo-0125"
    max_tokens = 1000
    max_turns = 3
    
    initial_messages_list, custom_id_list = load_harmbench_dataset()
    batch_processor = MultiTurnBatchProcessor(model, max_tokens, initial_messages_list, custom_id_list)
    
    for i in range(max_turns):
        batch_processor.execute_one_turn()
        # ! add your own answer to message list using add_messages function or manually
        # * example:
        for message in batch_processor.messages_list:
            message.append({"role": "user", "content": "Please answer the question"})
    
    batch_processor.save_messages()
        
        
