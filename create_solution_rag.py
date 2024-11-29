import json
import jsonlines

def read_jsonl(file_path):
    data = {}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            doc_id=obj['id']
            data[doc_id]=obj
    return data

with open('doc_id_to_poleval_id_mapping.json', 'r', encoding='utf-8') as f:
    doc_id_to_poleval_id = json.load(f)

# Create a mapping from poleval_id to doc_id
poleval_id_to_doc_id = {v: k for k, v in doc_id_to_poleval_id.items()}


def create_solution(path_in_tsv, path_rag, output_txt):

    rag = read_jsonl(path_rag)

    with open(output_txt, 'w', encoding='utf-8') as txtfile:
        for line in open(path_in_tsv, 'r', encoding='utf-8'):
            original_poleval_id = line.strip()
            poleval_id = original_poleval_id
            if original_poleval_id not in rag:
                if original_poleval_id in poleval_id_to_doc_id:
                    poleval_id = poleval_id_to_doc_id[original_poleval_id]
                else:
                    print(f"Warning: Poleval ID {original_poleval_id} not found in F1 or QA data")
                    txtfile.write('\n')
                    continue
            try:
                rag_answer = rag[poleval_id]['generated_answer']
            except:
                rag_answer = rag[original_poleval_id]['generated_answer']

            if rag_answer.lower().startswith('brak informacji'):
                rag_answer = ''
            txtfile.write(rag_answer + '\n')
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path_in_tsv", help="Path to the input TSV file")
    parser.add_argument("path_rag", help="Path to the F1 data file")
    parser.add_argument("output_txt", help="Path to the output txt file")
    args = parser.parse_args()
    
    create_solution(args.path_in_tsv, args.path_rag, args.output_txt)