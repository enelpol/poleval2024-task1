import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
from datasets import load_dataset
import jsonlines
import tqdm 
import Levenshtein
import re
def load_model_and_tokenizer(base_model_path, adapter_path):
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype='auto', device_map="auto")
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_answer(model, tokenizer, context, question, prompt_number, title, summary, max_new_tokens=50):
    prompts = [""]*7
    prompts[0] = f"Kontekst: {context}\nPytanie: {question}\nCzy kontekst jest relewantny dla pytania?\nOdpowiedź:"
    prompts[1] = f"Kontekst: {context}\nPytanie: {question}\nOdpowiedz krótko i zwięźle na powyższe pytanie.\nOdpowiedź:"
    prompts[2] = f"Kontekst: {context}\nPytanie: {question}\nJeśli kontekst zawiera odpowiedź na powyższe pytanie to odpowiedz krótko i zwięźle, a jeśli kontekst nie zawiera odpowiedzi to napisz: \"Brak informacji\".\nOdpowiedź: "
    prompts[3] = f"Tytuł: {title}\nKontekst: {context}\nPytanie: {question}\nOdpowiedz krótko i zwięźle na powyższe pytanie.\nOdpowiedź:"
    prompts[4] = f"Tytuł: {title}\nPodsumowanie: {summary}\nKontekst: {context}\nPytanie: {question}\nOdpowiedz krótko i zwięźle na powyższe pytanie.\nOdpowiedź:"
    prompts[5] = f"Tytuł: {title}\nKontekst: {context}\nPytanie: {question}\nJeśli kontekst zawiera odpowiedź na powyższe pytanie to odpowiedz krótko i zwięźle, a jeśli kontekst nie zawiera odpowiedzi to napisz: \"Brak informacji\".\nOdpowiedź: "
    prompts[6] = f"Tytuł: {title}\nKontekst: {context}\nPytanie: {question}\nJeśli kontekst zawiera odpowiedź na powyższe pytanie to odpowiedz krótko i zwięźle, a jeśli kontekst nie zawiera odpowiedzi to napisz: \"Brak informacji\".\nOdpowiedź:"
    

    inputs = tokenizer(prompts[prompt_number], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Odpowiedź:")[-1].strip()
    return answer

def main():
    parser = argparse.ArgumentParser(description="Run predictions with a LoRA-adapted language model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--prompt_number", type=int, required=True, help="Prompt number to be used")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--split", type=str, default='validation', help="Split to be used")
    args = parser.parse_args()

    raw_datasets = load_dataset('enelpol/poquad', trust_remote_code=True)
    eval_dataset = raw_datasets[args.split]

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter)
    
    levenshtein_norms = []
    # Iterate through the evaluation dataset
    with jsonlines.open(args.output_file, mode='w') as writer:
        for example in tqdm.tqdm(eval_dataset, total=len(eval_dataset)):
            context = example['context']
            question = example['question']
            title = example['title']
            summary = example['summary']
            if m:=re.search(r'\. [A-ZŁŃŚŻŹĆ]', summary):
                summary = summary[:m.start()]+'.'

            answer = generate_answer(model, tokenizer, context, question, args.prompt_number, title, summary)
            
            try:
                actual_answer = example['answers']['generative_answer'][0]
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
            'actual_answer': actual_answer
            })
            if actual_answer is not None:
                levenshtein_norm = 1-(Levenshtein.distance(answer.lower(), actual_answer.lower())/max(len(answer), len(actual_answer)))
                levenshtein_norms.append(levenshtein_norm)

                print(f"Average Levenshtein Normalized Distance: {sum(levenshtein_norms) / len(levenshtein_norms)}")

if __name__ == "__main__":
    main()
