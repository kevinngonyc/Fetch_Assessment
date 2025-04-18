# Kevin Ngo - Fetch Assessment
## Code
`notebook_draft.ipynb` is an early draft of the solutions <br/>
Final code for tasks may be found at `src/task_*.py`
### task_1.py
You may run `python src/task_1.py` to generate sample embeddings/similarities or run `python src/task_1.py [SENTENCE_1] [SENTENCE_2] ... [SENTENCE_N]` to generate custom embeddings/similarities.

### task_4.py
Run `python src/task_4.py` to train a multi-task learning model. Prints evaluation metrics every epoch.

### config.py
Various config settings.
```
sentence_transformer_model = "all-MiniLM-L6-v2"
sample_sentences = ["Her brother is a king.",
                     "His sister is a queen.",
                     "They can't dance."]
device = 'cuda'
epochs = 30
```

## Docker
Specify either '1' or '4' for the environment variable `$TASK` when running the Docker container
```
docker build -t IMAGE_NAME .
docker run -e TASK=[1|4] IMAGE_NAME
```

## Write-up
The write-up for tasks is in `writeup/writeup.txt`