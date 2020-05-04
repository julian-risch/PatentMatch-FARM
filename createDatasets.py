import pandas as pd
from sklearn.model_selection import train_test_split

path = "/mnt/data/datasets/patents/patent_matching"

positives =pd.read_csv(path+"/positives_satellite.csv",header=0)
negatives =pd.read_csv(path+"/negatives_satellite.csv",header=0)


positives = positives[['application_claim_text', 'patent_searchReport_paragraph']]
positives["label"]="1"

positives = positives.rename(columns={"application_claim_text": "text", "patent_searchReport_paragraph": "text_b"})
negatives = negatives[['application_claim_text', 'patent_searchReport_paragraph']]
negatives["label"]="0"
negatives = negatives.rename(columns={"application_claim_text": "text", "patent_searchReport_paragraph": "text_b"})

allSamples = positives.append(negatives)

samplesContainingNA_positive = positives[positives['text_b'].isnull()]
samplesContainingNA_negative = negatives[negatives['text_b'].isnull()]


allSamplesShuffled = allSamples.sample(frac=0.01)

train, test_dev = train_test_split(allSamplesShuffled, test_size=0.4)

test, dev = train_test_split(test_dev, test_size=0.5)

#print(len(allSamples))
#print(len(positives)/len(allSamples))
#print(len(negatives)/len(allSamples))

train.dropna().to_csv(path+"/train.tsv", sep="\t", index=False)
test.dropna().to_csv(path+"/test.tsv", sep="\t", index=False)
dev.dropna().to_csv(path+"/dev.tsv", sep="\t", index=False)

#samplesContainingNA_positive.to_csv(path+"/positives_NA.csv", sep=",", index=False)
#samplesContainingNA_negative.to_csv(path+"/negatives_NA.csv", sep=",", index=False)
'''
print("Total Positives:")
print(len(positives))
print("NA in Positives:")
print(len(samplesContainingNA_positive))
print("Correct in Positives:")
print(len(positives)-len(samplesContainingNA_positive))
print("Percentage of NA in Positives")
print(len(samplesContainingNA_positive)/len(positives))
print("\n")

print("Total Negatives:")
print(len(negatives))
print("NA in Negatives:")
print(len(samplesContainingNA_negative))
print("Correct Negatives:")
print(len(negatives)-len(samplesContainingNA_negative))
print("Percentage of NA in Negatives")
print(len(samplesContainingNA_negative)/len(negatives))
print("\n")

print("Distinct positive claim texts affected:")
print(samplesContainingNA_positive["text"].nunique())
print("Distinct negative claim texts affected:")
print(samplesContainingNA_negative["text"].nunique())
print("Distinct correct claims remaining:")
print(allSamples.dropna()["text"].nunique())
print("\n")

print("Correct overall:")
print(len(allSamples.dropna()))
print("Compare correct size for consistency...")
print(len(allSamples.dropna())==len(negatives)-len(samplesContainingNA_negative)+len(positives)-len(samplesContainingNA_positive))
'''