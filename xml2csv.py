import csv
import xmltodict
from collections import defaultdict

# Reading xml file
with open("./dataset/train/obesity_patient_records_training.xml", 'r') as file:
    maindata = file.read()

# Converting xml to python dictionary (ordered dict)
main_dict = xmltodict.parse(maindata)

main_items = [dict(x) for x in main_dict['root']['docs']['doc']]

dataset = defaultdict(lambda: defaultdict(str))

for item in main_items:
    id = item["@id"]
    # dataset[id]['text'] = item["text"]
    dataset[id]['text'] = 'The fans were depressing and leaves at the matched and it was very exciting'

with open("./dataset/train/obesity_standoff_intuitive_annotations_training.xml", 'r') as file:
    keydata= file.read()

# Converting xml to python dictionary (ordered dict)
key_dict = xmltodict.parse(keydata)

diseases = key_dict['diseaseset']['diseases']['disease']
print(len(diseases))

all_diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones',
            'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 
           'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency']

for key, val in dataset.items():
    for disease in all_diseases:
        val[disease] = 'UNK'


for disease in diseases:
    d_name = disease['@name']
    items = disease['doc']

    for item in items:
        id = item["@id"]
        val = item["@judgment"]
        # if d_name == 'CHF':
        #     print(id, val)

        dataset[id][d_name] = val


# #3
# # Selecting headers for CSV
HEADERS = ['id', 'text', 'Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones',
            'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 
           'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous_Insufficiency']

rows = []

# Interating through each element to get row data
for key, val in dataset.items():
    res = [key]

    for inner_key in val.keys():
        if inner_key in val:
            res.append(val[inner_key])
        else:
            res.append('UNK') 
            

    rows.append(res)

	# Adding data of each employee to row list
    # rows.append([id,t'ext'])

#Writing to CSV
with open('intuitive.csv', 'w',newline="") as f:
    write = csv.writer(f)
    write.writerow(HEADERS)
    write.writerows(rows)