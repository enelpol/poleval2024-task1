# Poleval 2024 Task 1

Run prediction using QA and answerability models:
```
python predict.py --base_model=speakleash/Bielik-11B-v2.0-Instruct --adapter=enelpol/poleval2024-task1-qa --prompt_number=1 --output_file=final_qa.jsonl --split=testB
python predict_seq.py --base_model=speakleash/Bielik-11B-v2.0-Instruct --adapter=enelpol/poleval2024-task1-answerability --prompt_number=3 --output_file=final_answerability.jsonl --split=testB
python create_solution.py data/test-B/in.tsv final_answerability.jsonl final_qa.jsonl test-B.tsv
```

Run prediction using RAG model:
```
python predict.py --base_model=speakleash/Bielik-11B-v2.0-Instruct --adapter=enelpol/poleval2024-task1-rag --prompt_number=5 --output_file=final_rag.jsonl --split=testB
python create_solution_rag.py data/test-B/in.tsv final_rag.jsonl rag-test-B.tsv
```