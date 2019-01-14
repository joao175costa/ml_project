from pipelines import cervical_cancer_classification, risk_screening_classification


# print results for the two implemented pipelines, with optimized steps
results = {}
print('Classification Results Using Cervical Cancer Pipeline')
for comb in ['', 'H', 'S', 'C', 'HS', 'HC', 'SC', 'HSC']:
    results[comb] = cervical_cancer_classification(screening=comb)
    print(comb, results[comb])

print('Classification Results Using 2-classifier Risk+Screening Pipeline')
results['2clf'] = risk_screening_classification()
print('2clf', results['2clf'])