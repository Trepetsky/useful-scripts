from tqdm import tqdm
import joblib
import requests, sys, json

WEBSITE_API = 'https://rest.uniprot.org'

def get_url(url, **kwargs):
    response = requests.get(url, **kwargs)

    if not response.ok:
        response.raise_for_status()
        sys.exit()

    return response

uniprot_data_proteins = {}
# proteins это датафрейм
for id in tqdm(proteins.UniProt.unique()):
    r = get_url(f'{WEBSITE_API}/uniprotkb/{id}')
    uniprot_data_proteins[str(id)] = r.json()

joblib.dump(uniprot_data_proteins, 'uniprot_data_proteins.pkl')
