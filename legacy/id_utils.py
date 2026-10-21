import os
import csv

from datetime import datetime
from pathlib import Path


def load_setup_dict() -> dict:
    with open('data_storage/experimental_setup.csv', newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # header skip
        return {int(row[0]): row[1].replace("['", '').replace("']", '') for row in list(reader)}


def get_input_by_id(id : int) -> tuple[str, str | None]:
    try:
        prompt = get_input_by_id.data[id]
        image_path = f"data_storage/RICO_dataset/{id}.jpg"
        use_transformers = os.getenv('USE_TRANSFORMERS', '0')
        if use_transformers == '0': image_path = None     # until ollama gets qwen2.5-vl support
        print(id, prompt)
        return prompt, image_path
    except AttributeError:
        get_input_by_id.data = load_setup_dict()
        return get_input_by_id(id)


def save_refined_prompts(refined_prompts : list[str], style_prompt : str, id_dir : str) -> None:
    with open(f"{id_dir}/prompts.txt", 'w') as file:
        file.write('\n'.join(refined_prompts))
        file.write('\n')
        file.write(style_prompt)
    return


def get_id_dir(output_dir : str, id : str | int) -> str:
    id_dir = f"{output_dir}/{id} {str(datetime.now()).replace(':', '-')}"
    Path(id_dir).mkdir(parents=True, exist_ok=True)
    return id_dir
