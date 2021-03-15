from django.shortcuts import render,request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece
from django.http import HttpResponse
import json

sentence = ""

def home(request):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]


    outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=120,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=2
    )


    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        data = json.dumps(line)
    
    return HttpResponse({'data':data}, content_type="application/json")

