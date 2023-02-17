

"""
    Code here was used to query the wikidata sequentially. This happened interactively, so code might not be entirely structured and sequential.

    UPDATE: I have updated the code to query wikidata and fetch the data. You can find that script in query_wikidata_new.py
    That should be used to query wikidata and fetch the data (this one will cause a lot of errors)
"""

import requests
import json
import pickle
from tqdm import tqdm
import time


TIMEOUT = 0.5
LANG = lang = ["EN", "FR", "ES", "ZH", "JA", "AR", "DE", "SV", "RU", "PT", "NL", "IT", "EL"]
LANG_FILTER = ['''langMatches( lang(?label), "{}" )'''.format(lan) for lan in LANG]


def getWikidata(query):
    endpointUrl = "https://query.wikidata.org/sparql"

    # The endpoint defaults to returning XML, so the Accept: header is required

    r = requests.get(endpointUrl, params={'query': query}, headers={'Accept': 'application/sparql-results+json'})
    print(r)
    if r.status_code == 429:
        # print("sleeping {} sec for 429 response".format(TIMEOUT))
        time.sleep(TIMEOUT)
        return getWikidata(query)
    data = r.json()
    print(data)
    label = data['results']['bindings'][0]['label']['value']
    return label


def generate_query(id):
    return (
        """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              PREFIX wd: <http://www.wikidata.org/entity/>
              SELECT  *
              WHERE {
                    wd:"""
        + id
        + """ rdfs:label ?label .
                    FILTER (langMatches( lang(?label), "EN" ) )
                  }
              LIMIT 1"""
    )


def generate_mapping(ids, start, end):

    '''Generate mapping as dictionary and export as pickle file'''



    mapping = {}
    for id in tqdm(ids[start:end]):
        # time.sleep(1)
        try:
            mapping[id] = getWikidata(generate_query(id))
        except IndexError:
            continue

    f = open("data/mapping/" + "{}_{}".format(start, end) + ".p", 'wb')
    print(mapping)
    pickle.dump(mapping, f)


def getWikidata_batch(query, ids):
    '''Query wikidata in batch'''
    endpointUrl = 'https://query.wikidata.org/sparql'

    # The endpoint defaults to returning XML, so the Accept: header is required
    r = requests.get(endpointUrl, params={'query': query}, headers={'Accept': 'application/sparql-results+json'})
    if r.status_code == 429:
        # print("sleeping {} sec for 429 response".format(TIMEOUT))
        time.sleep(TIMEOUT)
        return getWikidata_batch(query, ids)
    data = r.json()
    data = data['results']['bindings']
    ret = {}

    for item in data:
        id = item['id']['value'].split("/")[-1]
        label = item['label']['value']
        # description=item['description']['value']
        # ret[id]={"label":label, "description":description}
        ret[id] = label
    no_ans = []
    for id in ids:
        if id not in ret:
            no_ans.append(id)

    return ret, no_ans


def generate_query_batch(ids):
    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    # query = '''
    #   SELECT  distinct *
    #               WHERE {
    #                     VALUES ?id {''' + q + '''}
    #                     ?id rdfs:label ?label .
    #                 FILTER (langMatches( lang(?label), "EN" ) )
    #                     ?id schema:description ?description
    #                                  FILTER (langMatches( lang(?description), "EN" ) )
    #
    #                   } '''
    query = '''
          SELECT  distinct *
                      WHERE {
                            VALUES ?id {''' + q + '''}
                            ?id rdfs:label ?label .
                            
                        
                          } 
                          
                          '''
    return query


def remove_inv(id):
    if 'inv' in id:
        return id[:-4]
    return id


def generate_query_batch_relation(ids):
    ids = [remove_inv(id) for id in ids]

    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    query = '''
          SELECT  distinct *
                      WHERE {
                            VALUES ?id {''' + q + '''}
                            ?id rdfs:label ?label .
                        FILTER (langMatches( lang(?label), "EN" ) )
                            ?id schema:description ?description
                                         FILTER (langMatches( lang(?description), "EN" ) )
                          } '''
    return query


def generate_mapping_batch(ids, start, end):
    mapping = {}
    no_ans = []
    step = 400
    for i in tqdm(range(start, end, step)):
        ids_curr = ids[i:min(i + step, len(ids) - 1)]
        curr_mapping, curr_no_ans = getWikidata_batch(generate_query_batch(ids_curr), ids_curr)
        mapping.update(curr_mapping)
        no_ans += curr_no_ans
    # print(len(mapping))
    # print(len(no_ans))
    f = open("data/mapping/" + "{}_{}".format(start, end) + ".p", 'wb')

    pickle.dump(mapping, f)
    f1 = open("data/mapping/" + "{}_{}".format(start, end) + "_no_ans.p", 'wb')
    pickle.dump(no_ans, f1)


def generate_query_non_english(ids):
    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    query = '''
          SELECT  distinct *
                      WHERE {
                            VALUES ?id {''' + q + '''}
                            ?id rdfs:label ?label .
                        FILTER( langMatches( lang(?label), "EN" ) || langMatches( lang(?label), "FR" ) || langMatches( lang(?label), "ES" ) || langMatches( lang(?label), "ZH" ) || langMatches( lang(?label), "JA" ) || langMatches( lang(?label), "AR" ) || langMatches( lang(?label), "DE" ) || langMatches( lang(?label), "SV" ) || langMatches( lang(?label), "RU" ) || langMatches( lang(?label), "PT" ) || langMatches( lang(?label), "NL" ) || langMatches( lang(?label), "IT" ) || langMatches( lang(?label), "EL" ))
                            ?id schema:description ?description
                            FILTER( langMatches( lang(?description), "EN" ) || langMatches( lang(?description), "FR" ) || langMatches( lang(?description), "ES" ) || langMatches( lang(?description), "ZH" ) || langMatches( lang(?description), "JA" ) || langMatches( lang(?description), "AR" ) || langMatches( lang(?description), "DE" ) || langMatches( lang(?description), "SV" ) || langMatches( lang(?description), "RU" ) || langMatches( lang(?description), "PT" ) || langMatches( lang(?description), "NL" ) || langMatches( lang(?description), "IT" ) || langMatches( lang(?description), "EL" ))
                          } '''
    return query


def get_wikidata_non_english(query, ids):
    endpointUrl = 'https://query.wikidata.org/sparql'

    # The endpoint defaults to returning XML, so the Accept: header is required
    r = requests.get(endpointUrl, params={'query': query}, headers={'Accept': 'application/sparql-results+json'})
    if r.status_code == 429:
        # print("sleeping {} sec for 429 response".format(TIMEOUT))
        time.sleep(TIMEOUT)
        return get_wikidata_non_english(query, ids)
    # print(r)
    data = r.json()
    data = data['results']['bindings']
    ret = {}

    for item in data:
        id = item['id']['value'].split("/")[-1]
        label = item['label']['value']
        label_lang = item['label']['xml:lang']
        description = item['description']['value']
        des_lang = item['description']['xml:lang']
        if id in ret.keys():
            if label_lang in ret[id].keys():
                ret[id][label_lang]["label"] = label
            else:
                temp = {}
                temp["label"] = label
                ret[id][label_lang] = temp
            if des_lang in ret[id].keys():
                ret[id][des_lang]["description"] = description
            else:
                temp = {}
                temp["description"] = description
                ret[id][des_lang] = temp

        else:
            temp = {}
            temp[label_lang] = {"label": label}
            temp[des_lang] = {"description": description}
            ret[id] = temp

    no_ans = []
    for id in ids:
        if id not in ret:
            no_ans.append(id)

    return ret, no_ans


def generate_mapping_non_english(ids, start, end):
    mapping = {}
    no_ans = []
    step = 100
    for i in tqdm(range(start, end, step)):
        ids_curr = ids[i:min(i + step, len(ids) - 1)]
        curr_mapping, curr_no_ans = get_wikidata_non_english(generate_query_non_english(ids_curr), ids_curr)

        mapping.update(curr_mapping)

        no_ans += curr_no_ans

    f = open("data/mapping/" + "{}_{}_non_english".format(start, end) + ".p", 'wb')

    pickle.dump(mapping, f)
    f1 = open("data/mapping/" + "{}_{}".format(start, end) + "_no_ans_non_english.p", 'wb')
    pickle.dump(no_ans, f1)
    return no_ans


def generate_query_instance(ids):
    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    query = '''
              SELECT  distinct *
                          WHERE {
                                VALUES ?id {''' + q + '''}
                                
                                ?id wdt:P31 ?instance
                              } '''
    return query

def generate_query_labels(ids):
    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    query = '''
                  SELECT  distinct *
                              WHERE {
                                    VALUES ?id {''' + q + '''}
                                    ?id rdfs:label ?label .
                                FILTER( langMatches( lang(?label), "EN" ) || langMatches( lang(?label), "FR" ) || langMatches( lang(?label), "ES" ) || langMatches( lang(?label), "ZH" ) || langMatches( lang(?label), "JA" ) || langMatches( lang(?label), "AR" ) || langMatches( lang(?label), "DE" ) || langMatches( lang(?label), "SV" ) || langMatches( lang(?label), "RU" ) || langMatches( lang(?label), "PT" ) || langMatches( lang(?label), "NL" ) || langMatches( lang(?label), "IT" ) || langMatches( lang(?label), "EL" ))
                                  } '''
    return query

def generate_query_descriptions(ids):
    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    query = '''
              SELECT  distinct *
                          WHERE {
                                VALUES ?id {''' + q + '''}
                                
                                ?id schema:description ?description
                                FILTER( langMatches( lang(?description), "EN" ) || langMatches( lang(?description), "FR" ) || langMatches( lang(?description), "ES" ) || langMatches( lang(?description), "ZH" ) || langMatches( lang(?description), "JA" ) || langMatches( lang(?description), "AR" ) || langMatches( lang(?description), "DE" ) || langMatches( lang(?description), "SV" ) || langMatches( lang(?description), "RU" ) || langMatches( lang(?description), "PT" ) || langMatches( lang(?description), "NL" ) || langMatches( lang(?description), "IT" ) || langMatches( lang(?description), "EL" ))
                              } '''
    return query







def generate_query_sameAs(ids):
    q = ["wd:" + ids[i] for i in range(len(ids))]
    q = " ".join(q)
    query = '''
                  SELECT  distinct *
                              WHERE {
                                    VALUES ?id {''' + q + '''}
                                    ?id owl:sameAs ?target
                                  } '''
    return query


def get_wikidata_sameAs(query, ids):
    endpointUrl = 'https://query.wikidata.org/sparql'

    # The endpoint defaults to returning XML, so the Accept: header is required
    r = requests.get(endpointUrl, params={'query': query}, headers={'Accept': 'application/sparql-results+json'})
    if r.status_code == 429:
        # print("sleeping {} sec for 429 response".format(TIMEOUT))
        time.sleep(TIMEOUT)
        return get_wikidata_sameAs(query, ids)
    data = r.json()
    data = data['results']['bindings']
    ret = {}
    for item in data:
        id = item['id']['value'].split("/")[-1]
        sameAsID = item['target']['value'].split("/")[-1]
        ret[id] = sameAsID
    no_ans = []

    for id in ids:
        if id not in ret:
            no_ans.append(id)
    return ret, no_ans


def generate_mapping_sameAs(ids, start, end):
    mapping = {}
    no_ans = []
    step = 400

    for i in tqdm(range(start, end, step)):
        ids_curr = ids[i:min(i + step, len(ids) - 1)]
        curr_mapping, curr_no_ans = get_wikidata_sameAs(generate_query_sameAs(ids_curr), ids_curr)
        # print(curr_mapping)
        # print(sum([("label" in item.keys() or "description" in item.keys()) for item in curr_mapping.values()]))

        mapping.update(curr_mapping)

        no_ans += curr_no_ans

    f = open("data/mapping/" + "{}_{}_sameAs".format(start, end) + ".p", 'wb')

    pickle.dump(mapping, f)
    f1 = open("data/mapping/" + "{}_{}".format(start, end) + "_no_ans_sameAs.p", 'wb')
    pickle.dump(no_ans, f1)
    return no_ans


def get_wikidata_instance(query, ids):
    endpointUrl = 'https://query.wikidata.org/sparql'

    # The endpoint defaults to returning XML, so the Accept: header is required
    r = requests.get(endpointUrl, params={'query': query}, headers={'Accept': 'application/sparql-results+json'})
    if r.status_code == 429:
        # print("sleeping {} sec for 429 response".format(TIMEOUT))
        time.sleep(TIMEOUT)
        return get_wikidata_instance(query, ids)
    # print(r)
    data = r.json()
    data = data['results']['bindings']
    ret = {}
    instance_ids = set()
    for item in data:
        id = item['id']['value'].split("/")[-1]
        instance_id = item['instance']['value'].split("/")[-1]
        ret[id] = {"instance": instance_id}
        instance_ids.add(instance_id)

    no_ans = []

    for id in ids:
        if id not in ret:
            no_ans.append(id)

    return ret, no_ans, instance_ids


def generate_mapping_instance(ids, start, end):
    mapping = {}
    no_ans = []
    step = 400
    instance_ids = set()
    for i in tqdm(range(start, end, step)):
        ids_curr = ids[i:min(i + step, len(ids) - 1)]
        curr_mapping, curr_no_ans, id = get_wikidata_instance(generate_query_instance(ids_curr), ids_curr)
        # print(curr_mapping)
        # print(sum([("label" in item.keys() or "description" in item.keys()) for item in curr_mapping.values()]))

        mapping.update(curr_mapping)

        no_ans += curr_no_ans
        instance_ids.update(id)

    f = open("data/mapping/" + "{}_{}_instance".format(start, end) + ".p", 'wb')

    pickle.dump(mapping, f)
    f1 = open("data/mapping/" + "{}_{}".format(start, end) + "_no_ans_instance.p", 'wb')
    pickle.dump(no_ans, f1)
    return no_ans, instance_ids


"""
    Note: The code below was used in iterations to download the all of the wikidata information. 
"""
from googletrans import Translator
from tqdm import tqdm
from wikidata.client import Client
from googletrans import Translator

def get_instances():
    no_ans=[]
    base=0
    instance_ids=set()
    path="data/ent2ids"
    data=json.load(open(path,'r'))
    ids=list(data.keys())
    for i in range(484):
        noans, id= generate_mapping_instance(ids, base+i*400*25, base+(i+1)*400*25)
        no_ans+=noans
        instance_ids.update(id)

    f=open("data/mapping/no_instance_ids.p",'wb')
    pickle.dump(no_ans,f)
    f=open("data/mapping/instance_ids.p",'wb')
    pickle.dump(instance_ids,f)

    instance_id_mapping=pickle.load(open("data/Mapping/instance_name.p",'rb'))
    english_map={}
    non_english_map={}
    instance={}
    for i in tqdm(range(484)):
        temp = pickle.load(open("data/Mapping/{}_{}_english.p".format(i*10000,(i+1)*10000),"rb"))
        temp2= pickle.load(open("data/Mapping/{}_{}_instance.p".format(i*10000,(i+1)*10000),"rb"))
        english_map.update(temp)
        for id in temp2.keys():
            temp2[id]={'instance':instance_id_mapping[temp2[id]['instance']]}
        instance.update(temp2)

    for i in tqdm(range(144)):
        temp2 = pickle.load(open("data/Mapping/{}_{}_non_english.p".format(i * 10000, (i + 1) * 10000), "rb"))
        non_english_map.update(temp2)

    for key in tqdm(instance.keys()):
        if key in english_map:
            instance[key].update(english_map[key])
        if key in non_english_map:
            instance[key].update(non_english_map[key])

    f=open("data/Mapping/total_entity.p",'wb')
    pickle.dump(instance,f)


ids=list(pickle.load(open("data/Mapping/no_instance_name2.p",'rb')))


translator=Translator()
mapping={}
no_ans=[]
step=20
for i in tqdm(range(0,len(ids),step)):
    ids_curr=ids[i:min(i+step, len(ids)-1)]
    curr_mapping,curr_no_ans=getWikidata_batch(generate_query_batch(ids_curr),ids_curr)
    for key in tqdm(curr_mapping.keys()):
        time.sleep(1)

        label=translator.translate(curr_mapping[key]).text
        mapping[key]=label
    no_ans+=curr_no_ans


    mapping.update(curr_mapping)
    no_ans+=curr_no_ans
# # print(len(mapping))
# # print(len(no_ans))
f=open("data/mapping/translated2.p", 'wb')
pickle.dump(mapping, f)
mapp=pickle.load(open("data/mapping/instance_name2.p",'rb'))
mapp.update(mapping)
f=open("data/mapping/instance_name3.p", 'wb')
pickle.dump(mapp, f)
f1=open("data/mapping/no_instance_name3.p", 'wb')
pickle.dump(no_ans,f1)

ids=list(pickle.load(open("data/Mapping/instance_ids.p",'rb')))
mapping={}
no_ans=[]
step=400
for i in tqdm(range(0,len(ids),step)):
    ids_curr=ids[i:min(i+step, len(ids)-1)]
    curr_mapping,curr_no_ans=getWikidata_batch(generate_query_batch(ids_curr),ids_curr)
    mapping.update(curr_mapping)
    no_ans+=curr_no_ans


#     mapping.update(curr_mapping)
#     no_ans+=curr_no_ans
# # print(len(mapping))
# # print(len(no_ans))

f=open("data/mapping/instance_name.p", 'wb')
pickle.dump(mapping, f)
f1=open("data/mapping/no_instance_name.p", 'wb')
pickle.dump(no_ans,f1)


ids=[]
for i in range(484):
    temp=pickle.load(open("data/mapping/{}_{}_no_ans.p".format(i*10000, (i+1)*10000),'rb'))
    ids+=temp

no_ans=[]
base=0
for i in range(144):
    no_ans+=generate_mapping_non_english(ids,base+i*200*50, base+(i+1)*200*50)
f=open("data/mapping/phase3IDs.p",'wb')
pickle.dump(no_ans,f)



path="data/ent2ids"
data=json.load(open(path,'r'))
ids=list(data.keys())
base=7700
s=ids[0+base:400+base]
q=["wd:"+s[i] for i in range(len(s))]
q=" ".join(q)


for i in range(200):
    generate_mapping(ids,base+i*100,base+(i+1)*100)
#getWikidata(generate_query("Q36079160"))
query='''
  SELECT  distinct *
              WHERE {
                    VALUES ?id {'''+q+'''}
                    ?id rdfs:label ?label .
                FILTER (langMatches( lang(?label), "EN" ) )
                    ?id schema:description ?description
                                 FILTER (langMatches( lang(?description), "EN" ) )
                  } '''

mapping, noans=getWikidata_batch(query,s)
print(mapping)
print(noans)
base=4840000
print(len(ids))
print(ids[base:base+400])
for i in range(52):
    generate_mapping_batch(ids,base+i*400*25, base+(i+1)*400*25)
path="data/relation2ids"
data=json.load(open(path,'r'))
ids=list(data.keys())

generate_mapping_batch(ids,0,1600)
query = '''SELECT  distinct *
              WHERE {
                    VALUES ?id {wd:Q16521405 wd:Q11510006 wd:Q18839780 wd:Q45043472}
                    ?id rdfs:label ?label .
                    ?id wdt:P31 ?instance
                  } '''
getWikidata(query)


ids = pickle.load(open("data/Mapping/no_instance_ids.p", 'rb'))
no_ans = generate_mapping_sameAs(ids, 0, len(ids))
print(len(no_ans))

def find_without_instanceof():
    """
        Find those without instanceof relationship
        Could be either: Redirects, deleted ones or other
        All left are deleted ones
    """
    ids=pickle.load(open("data/Mapping/no_instance_ids.p",'rb'))

    from wikidata.client import Client
    client = Client()

    for i in range(28):
        map = {}
        no_ans = []
        _ids=ids[i*2000:(i+1)*2000]
        for id in tqdm(_ids):
            try:
                entity=client.get(id,load=True)
                map[id] = entity.__dict__["data"]
            except:
                no_ans.append(id)
                continue


        pickle.dump(map,open("data/Mapping/{}_sameAs_dict.p".format(i),'wb'))
        pickle.dump(no_ans,open("data/Mapping/{}_no_ans_sameAs_dict.p".format(i),'wb'))

client = Client()

translator=Translator()
map=pickle.load(open("data/Mapping/total_entity.p",'rb'))
update_map = {}
for i in range(28):

    dic=pickle.load(open("data/Mapping/{}_sameAs_dict.p".format(i),'rb'))
    for key in tqdm(dic.keys()):
        update_map[key]={}
        if 'P31' in dic[key]['claims']:
            id=dic[key]['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']
            instance=client.get(id,load=True)
            if 'en' in instance.label.texts:
                name=instance.label.texts['en']
            else:
                name=translator.translate(str(instance.label)).text
            update_map[key]['instance']=name


        labels=dic[key]['labels']
        for language in labels.keys():
            if language in update_map[key]:
                update_map[key][language]['label']=labels[language]
            else:
                update_map[key][language]={'label':labels[language]}

        des=dic[key]['descriptions']
        for language in des.keys():
            if language in update_map[key]:
                update_map[key][language]['description']=des[language]
            else:
                update_map[key][language]={'description':des[language]}

    pickle.dump(update_map,open("data/Mapping/update_map.p",'wb'))
    map.update(update_map)
    pickle.dump(map,open("data/Mapping/total_entity2.p",'wb'))

    data=pickle.load(open("data/Mapping/update_map.p",'rb'))
    for key in data:
        for lan in data[key]:
            if lan!='instance':
                for name in data[key][lan]:
                    data[key][lan][name]=data[key][lan][name]['value']
    pickle.dump(data,open("data/Mapping/update_map2.p",'wb'))





def gen_mapping(start, end):
    f = open("data/Wiki/mapping/" + "{}_{}".format(start, end) + ".p", "wb")
    print(mapping)
    pickle.dump(mapping, f)


    path = "data/Wiki/ent2ids"
    data = json.load(open(path, "r"))
    ids = list(data.keys())


    for i in range(200):
        generate_mapping(ids, i * 100, (i + 1) * 100)