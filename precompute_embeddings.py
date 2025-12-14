from transformers import AutoTokenizer, AutoModel
import torch, json

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_model = AutoModel.from_pretrained(model_name)
text_model.eval()

with open("ds005170/json/textmaps.json", "r", encoding="utf-8") as f:
    textmaps = json.load(f)          # {sentence: class_id}

sentences = list(textmaps.keys())
sent_embs = {}

with torch.no_grad():
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt", padding=True, truncation=True)
        outputs = text_model(**inputs)
        # simple choice: CLS token
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (hidden_size,)
        sent_embs[s] = cls_emb.tolist()

with open("ds005170/json/sentence_bert_embs.json", "w", encoding="utf-8") as f:
    json.dump(sent_embs, f, ensure_ascii=False)