
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from torch_utils import *

to_str = {0: 'AE',
 1: 'BH',
 2: 'DZ',
 3: 'EG',
 4: 'IQ',
 5: 'JO',
 6: 'KW',
 7: 'LB',
 8: 'LY',
 9: 'MA',
 10: 'OM',
 11: 'PL',
 12: 'QA',
 13: 'SA',
 14: 'SD',
 15: 'SY',
 16: 'TN',
 17: 'YE'}

def load_dl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('../models/arabert_arabic_dialect.pth',  map_location=device)
    model.to(device)
    model.eval()
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-medium-arabic')
    return tokenizer

class PredictDataset(Dataset):
    def __init__(self,data,max_len, tokenizer):
        super().__init__()
        self.texts = data["text"].values
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = " ".join(self.texts[idx].split())
        inputs = self.tokenizer(text,padding='max_length',
                                max_length=self.max_len,truncation=True,return_tensors="pt")
        #input_ids,token_type_ids,attention_mask
        return {
            "inputs":{"input_ids":inputs["input_ids"][0],
                      "token_type_ids":inputs["token_type_ids"][0],
                      "attention_mask":inputs["attention_mask"][0],
                     },
        }

def predict_dialect(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PredictDataset(text, 70, tokenizer)
    dataloder = DataLoader(dataset,batch_size=8,shuffle=False)
    
    preds = []
    for batch in dataloder:    
        x = batch["inputs"]
        inp = {k: v.to(device) for k, v in x.items()}
        
        with torch.no_grad():
            outputs = model(inp)

        predictions = torch.argmax(outputs, dim=1)
        
        preds.extend(predictions)
    
    return [to_str[i.item()] for i in preds]