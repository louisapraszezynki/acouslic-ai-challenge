from datasets import load_dataset

entire_dataset = load_dataset("Louloubib/acouslic_ai")
new_dataset = entire_dataset["train"].filter(lambda e: e['label'] in [1, 2])

new_dataset.push_to_hub('acouslic_ai_filtered_for_segmentation', token='***')