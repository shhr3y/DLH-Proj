morbidities = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous_Insufficiency']

for m in morbidities:
    with open(f'{m.lower()}.txt', 'w') as f:
        f.write(f'============================{m.upper()}============================')