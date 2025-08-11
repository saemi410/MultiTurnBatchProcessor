# 🔄 MultiTurnBatchProcessor

A Python utility for running **multi-turn conversations** with the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch?lang=python). It automates batch creation → upload → status polling → result retrieval → message history updates — all in one flow.

### ✨ Features
- Multi-turn conversation tracking
- `.jsonl` batch file generation
- Automatic upload & polling
- Timestamped log storage


### 📦 Requirements
```
pip install openai
export OPENAI_API_KEY="your_api_key"
```

### 🛠 Usage

1. **Prepare initial messages**
- Your data should be a list of conversation histories, e.g.:
```
initial_messages = [
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Tell me a fun fact about space."}],
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Explain quantum entanglement simply."}]
]
custom_ids = ["req-1", "req-2"]
```

2. **Create the processor**
```
bp = MultiTurnBatchProcessor(
    model="gpt-3.5-turbo-0125",
    max_tokens=1000,
    initial_messages=initial_messages,
    custom_id_list=custom_ids
)
```

3. **Run multiple turns**
```
for _ in range(3):  # number of turns
    bp.execute_one_turn()
    # Append your own follow-up question for next turn
    for msg in bp.messages_list:
        msg.append({"role": "user", "content": "Please elaborate."})
```

4. **Save results**
```
bp.save_messages()
```
### 📂 Output
- `logs/{model}/{timestamp}/turnN.jsonl` – batch input files
- `logs/{model}/{timestamp}/messages_list.json` – final conversation history

### Notes
The repository includes an example CSV loader using the HarmBench dataset, but you can replace it with any dataset or prompt source. If you encounter any issues, feel free to reach out anytime! 🤝