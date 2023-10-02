import pandas as pd
data = pd.read_csv('../data/output_scores.csv')
print(data)

count = 0
for idx in data.index:
	if data.scores[idx] > .5:
		count += 1
		print(data.scores[idx])
		print(data.sentence1[idx])
		print('\n')
		print(data.sentence2[idx])
print(count)
