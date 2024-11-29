import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
from datasets import load_dataset
import jsonlines
import tqdm 

def load_model_and_tokenizer(base_model_path, adapter_path):
    # Load the base model
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2, torch_dtype='auto', device_map="auto")
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    #tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return model, tokenizer

def generate_answer(model, tokenizer, context, question, prompt_number, title, summary, max_new_tokens=50):
    prompts = [""]*5
    prompts[0] = f"Kontekst: {context}\nPytanie: {question}\nCzy kontekst jest relewantny dla pytania?\nOdpowiedź: " #+tokenizer.eos_token
    prompts[1] = f"Kontekst: {context}\nPytanie: {question}\nOdpowiedz krótko i zwięźle na powyższe pytanie.\nOdpowiedź:"
    prompts[2] = f"Kontekst: {context}\nPytanie: {question}\nJeśli kontekst zawiera odpowiedź na powyższe pytanie to odpowiedz krótko i zwięźle, a jeśli kontekst nie zawiera odpowiedzi to napisz: \"Brak informacji\".\nOdpowiedź: "
    prompts[3] = f"Tytuł: {title}\nKontekst: {context}\nPytanie: {question}\nCzy kontekst jest relewantny dla pytania?\nOdpowiedź: "
    prompts[4] = f"Tytuł: {title}\nPodsumowanie: {summary}\nKontekst: {context}\nPytanie: {question}\nCzy kontekst jest relewantny dla pytania?\nOdpowiedź: "
     

    inputs = tokenizer(prompts[prompt_number], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        #print(inputs)
        outputs = model(**inputs)
        logits = outputs.logits
        #print(logits)
        predicted_class = torch.argmax(logits, dim=1).item() #1 means positive (impossible)
    
        probabilities = torch.softmax(logits, dim=1)
        probability = probabilities[0][predicted_class].item()

    return "Tak" if predicted_class == 1 else "Nie", probability

def main():
    parser = argparse.ArgumentParser(description="Run predictions with a LoRA-adapted sequence classification model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--prompt_number", type=int, required=True, help="Prompt number to be used")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset before processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to process")
    parser.add_argument("--split", type=str, default='validation', help="Split to be used")
    args = parser.parse_args()

    raw_datasets = load_dataset('enelpol/poquad', trust_remote_code=True)
    eval_dataset = raw_datasets[args.split]

    if args.shuffle:
        eval_dataset=eval_dataset.shuffle(seed=args.seed)
    if args.limit:
        eval_dataset=eval_dataset.select(range(args.limit))

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter)
    
    # Iterate through the evaluation dataset
    i=0
    tp=fp=fn=tn=0
    with jsonlines.open(args.output_file, mode='w') as writer:
        for example in tqdm.tqdm(eval_dataset, total=len(eval_dataset)):
            context = example['context']
            question = example['question']
            title = example['title']
            summary = example['summary']
            


            answer, probability = generate_answer(model, tokenizer, context, question, args.prompt_number, title, summary)
            try:
                actual_answer = 'Tak' if example['is_impossible'] else 'Nie'
            except:
                actual_answer = None
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Actual Answer: {actual_answer}")
            print("---")
            writer.write({
            'question': question,
            'context': context,
            'is_impossible': example['is_impossible'],
            'id': example['id'],
            'generated_answer': answer,
            'actual_answer': actual_answer,
            'probability': probability
            })
            i+=1
            
            if actual_answer is not None:
                if answer == 'Tak' and  actual_answer == 'Tak':
                    tp+=1
                elif answer == 'Nie' and actual_answer == 'Nie':
                    tn+=1
                elif answer == 'Tak' and actual_answer == 'Nie':
                    fp+=1
                elif answer == 'Nie' and actual_answer == 'Tak':
                    fn+=1
                
                #if i==10:
                #    break

                precision = tp/(tp+fp) if tp+fp>0 else 0    
                recall = tp/(tp+fn) if tp+fn>0 else 0
                f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
                print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
                print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

if __name__ == "__main__":
    main()
