# pip install biopython
# pip install tensor_parallel
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import torch
from tqdm import tqdm
import tensor_parallel as tp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").eval().to(device)
model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])
fasta_file = 'file.fasta'

sequences = [" ".join(str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")]

batch_size = 16
embeddings = torch.zeros((len(sequences), 1024), device=device)

for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i + batch_size]
        actual_batch_size = len(batch)
        encoded_input = tokenizer(batch, return_tensors='pt', padding=True, max_length=512, truncation=True).to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            batch_embeddings = model_output[0][::,0,::]
            embeddings[i:i+actual_batch_size] = batch_embeddings
