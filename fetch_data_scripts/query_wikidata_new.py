import pandas as pd
from wikidata.client import Client
from tqdm import tqdm
from collections import defaultdict
import requests
import time
import urllib


TIMEOUT = 0.5
BATCHSIZE = 400
LANGUAGE_PRIORITY = dict([(x.lower(), i) if x.lower()[:2] != "en" else (x.lower(),0) for i, x in enumerate(["EN", "EN-CA", "EN-GB", "EN-AU",  "SV", "DE", "ES", "PT", "PT-BR", "NL", "IT", "FR", "ZH", "JA", "AR", "RU", "EL"])])
SERVICEURL = "https://query.wikidata.org/sparql"


def batchfetch_WD(query, ids, which_ones : str = ["label", "description", "instance_of"]):
    r = requests.get(SERVICEURL, params={'query': query}, headers={'Accept': 'application/sparql-results+json'})
    
    if r.status_code == 429:
        time.sleep(TIMEOUT)
        return batchfetch_WD(query, ids, which_ones=which_ones)
    data = r.json()['results']['bindings']
    ret = defaultdict(dict)
    
    for res_entry in data:
        idx = res_entry['id']['value'].split("/")[-1]
        if "label" not in ret[id]:
            ret[idx]["label"] = []
        if "description" not in ret[id]:
            ret[idx]["description"] = []

        for which in which_ones:
            if which in res_entry:
                ret[idx][which].append((res_entry[which]["xml:lang"][:2], res_entry[which]['value']))
        
    for id in ret:
        for which in which_ones:
            if len(ret[id][which]) == 0:
                ret[id][which] = "Unknown"
            else:
                ret[id][which] = sorted(ret[id][which], key=lambda x: LANGUAGE_PRIORITY[x[0].lower()])[0][1]
    nullresp = []
    for id in ids:
        if id not in ret:
            nullresp.append(id)

    return ret, nullresp






def main(args):



    df = pd.read_csv(args.datapath,header=None)
    df.columns = ["head", "relation", "tail"]

    unique_entities = sorted(set(df['head'].unique()).union(set(df['tail'].unique())))
    unique_relations = sorted(set(df['relation'].unique()))

    
    client = Client()
    res = defaultdict(dict)
    if args.skip_rels is False:
        for rel in tqdm(unique_relations):
            try:

                rel = client.get(rel, load=True)
                res[rel.id]["label"] = rel.label
                res[rel.id]["description"] = rel.description
            except urllib.error.HTTPError:
                res[rel]["label"] = "Unknown"
                res[rel]["description"] = "Unknown"
            
        df = pd.DataFrame.from_dict(res, orient='index')
        df.to_csv("relation_descriptions.csv")
    res = defaultdict(dict)
    with open("entity_descriptions_buffer.csv", "w+") as f:
        f.write("id,label,description\n")
        for entitites in tqdm(range(0, len(unique_entities), BATCHSIZE)):
            ids = unique_entities[entitites:(entitites + BATCHSIZE)]
            query = ["wd:" + ids[i] for i in range(len(ids))]
            query = " ".join(query)
            query = '''SELECT distinct * WHERE { VALUES ?id {''' + query + '''} ?id rdfs:label ?label . FILTER (langMatches( lang(?label), "EN" ) ) ?id schema:description ?description FILTER (langMatches( lang(?description), "EN" ) ||  langMatches( lang(?description), "SV" ) || langMatches( lang(?description), "PT") ) } '''
            try:
                results, nullresponse = batchfetch_WD(query, ids)
            
            
                for result in results:
                    res[result]["label"] = results[result]["label"]
                    res[result]["description"] = results[result]["description"]
                    f.write("{},{},{}\n".format(result, results[result]["label"], results[result]["description"]))
                for id in nullresponse:
                    res[id]["label"] = "Unknown"
                    res[id]["description"] = "Unknown"
                    f.write("{},{},{}\n".format(id, "Unknown", "Unknown"))
            except:
                for id in ids:
                    f.write("{},{},{}\n".format(id, "Error", "Error"))
        
    df = pd.DataFrame.from_dict(res, orient='index')
    df.to_csv("entities_descriptions.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rels", action="store_true")
    parser.add_argument("--datapath", type=str, default="alltriples.txt")
    args = parser.parse_args()
    main(args)


