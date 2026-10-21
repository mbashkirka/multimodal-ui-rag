from faiss import read_index
from PIL import Image
import os
import clip
import json
import torch
from transformers import CLIPTokenizer

from utils.id_utils import get_input_by_id, get_id_dir


class App:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.preprocess = preprocess
        self.model.eval()

        self.index = read_index("data_storage/faiss_index.faiss")
        with open("data_storage/image_index_map.json") as f:
            self.image_paths = json.load(f)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    def encode_image(self, image_path):
        image = Image.open(image_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()


    def encode_text(self, search_text):
        max_len = 77
        stride = 30

        words = search_text.split()
        segments_text = []
        for i in range(0, len(words), stride):
            segment = " ".join(words[i:i + max_len - 2])
            segments_text.append(segment)

        segments = []
        for segment_text in segments_text:
            tokens = self.tokenizer(segment_text, truncation=True, padding="max_length", max_length=max_len-2)["input_ids"]

            cls_token = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id or 0
            sep_token = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 0
            pad_token = self.tokenizer.pad_token_id or 0

            segment = [cls_token] + tokens[:max_len-2] + [sep_token]
            segment += [pad_token] * (max_len - len(segment))

            segments.append(segment)

        embeddings = []
        for segment in segments:
            text_inputs = torch.tensor([segment]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs).float()
            embeddings.append(text_features)

        final_embedding = torch.mean(torch.stack(embeddings), dim=0)
        final_embedding /= final_embedding.norm(dim=-1, keepdim=True)

        return final_embedding.cpu().numpy()


    def search(self, search_text, image_path=None, results=1):
        text_features = self.encode_text(search_text)

        if image_path:
            image_features = self.encode_image(image_path)
            combined_features = (text_features + image_features) / 2
        else:
            combined_features = text_features

        _, indices = self.index.search(combined_features, results)
        image_base_path = "data_storage/RICO_dataset/"

        return [image_base_path + self.image_paths[str(indices[0][i])] for i in range(results)]


    def run(self, text_prompt : str, reference_image_path : str | None, screens : int, output_dir : str):
        results = self.search(text_prompt, reference_image_path, screens)
        os.makedirs(output_dir, exist_ok=True)
        for i, image_path in enumerate(results):
            try:
                image = Image.open(image_path)
                output_path = os.path.join(output_dir, f"{i+1}.png")
                image.save(output_path)
            except Exception as e:
                print(e, "Make sure you placed Rico dataset in the data_storage/RICO_dataset")
        return


def retrieval_baseline(
    id : str | int,
    reference_image_path : str | None = None,
    screens : int = 5,
    output_dir : str = "output_data/retrieval_baseline"
) -> str:
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    if type(id) == str:
        prompt = id
    else:
        prompt, _ = get_input_by_id(id)

    id_dir = get_id_dir(output_dir, id)

    app = App()
    app.run(prompt, reference_image_path, screens, id_dir)

    return id_dir
