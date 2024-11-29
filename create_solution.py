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


def create_solution(path_in_tsv, path_f1, path_qa, output_txt):
    # Load F1 data
    f1 = read_jsonl(path_f1)
    qa = read_jsonl(path_qa)

    with open(output_txt, 'w', encoding='utf-8') as txtfile:
        for line in open(path_in_tsv, 'r', encoding='utf-8'):
            original_poleval_id = line.strip()
            poleval_id = original_poleval_id
            if original_poleval_id not in f1 and original_poleval_id not in qa:
                if original_poleval_id in poleval_id_to_doc_id:
                    poleval_id = poleval_id_to_doc_id[original_poleval_id]
                else:
                    print(f"Warning: Poleval ID {original_poleval_id} not found in F1 or QA data")
                    txtfile.write('\n')
                    continue
            try:
                f1_answer = f1[poleval_id]['generated_answer']
            except:
                f1_answer = f1[original_poleval_id]['generated_answer']
            qa_answer = qa[poleval_id]['generated_answer']
            #print(answer)
            assert f1_answer in ['Tak', 'Nie']
            answer = qa_answer if f1_answer == 'Nie' else ''
            txtfile.write(answer + '\n')
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path_in_tsv", help="Path to the input TSV file")
    parser.add_argument("path_f1", help="Path to the F1 data file")
    parser.add_argument("path_qa", help="Path to the QA data file")
    parser.add_argument("output_txt", help="Path to the output txt file")
    args = parser.parse_args()
    
    create_solution(args.path_in_tsv, args.path_f1, args.path_qa, args.output_txt)