import pandas as pd
from sklearn.model_selection import train_test_split

def execute():
    path = "/mnt/data/datasets/patents/patent_matching"
    positives =pd.read_csv(path+"/positives_satellite.csv",header=0)
    negatives =pd.read_csv(path+"/negatives_satellite.csv",header=0)
    sample_size=0.01


    positives = positives[['application_claim_text', 'patent_searchReport_paragraph']]
    positives["label"]="1"

    positives = positives.rename(columns={"application_claim_text": "text", "patent_searchReport_paragraph": "text_b"})
    negatives = negatives[['application_claim_text', 'patent_searchReport_paragraph']]
    negatives["label"]="0"
    negatives = negatives.rename(columns={"application_claim_text": "text", "patent_searchReport_paragraph": "text_b"})

    allSamples = positives.append(negatives).dropna()

    #Remove "</p" at the end of paragraphs
    allSamples['text_b']=allSamples['text_b'].str.replace('<\/p','',regex=True)
    allSamples['text'] = allSamples['text'].str.replace('<\/p', '', regex=True)
    #Remove everything written in "<...>"
    allSamples['text_b']=allSamples['text_b'].str.replace('\<.+?\>', '', regex=True)
    allSamples['text'] = allSamples['text'].str.replace('\<.+?\>', '', regex=True)

    #samplesContainingNA_positive = positives[positives['text_b'].isnull()]
    #samplesContainingNA_negative = negatives[negatives['text_b'].isnull()]

    train,test,dev=train_test_dev_split(allSamples,0.4,0.3,0.3)

    #Shuffle and sample from data
    train = allSamples.sample(frac=sample_size)
    test = test.sample(frac=sample_size)
    dev = dev.sample(frac=sample_size)





    #print(len(allSamples))
    #print(len(positives)/len(allSamples))
    #print(len(negatives)/len(allSamples))

    train.to_csv(path+"/train.tsv", sep="\t", index=False)
    test.to_csv(path+"/test.tsv", sep="\t", index=False)
    dev.to_csv(path+"/dev.tsv", sep="\t", index=False)

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


def train_test_dev_split(samples, train_size=0.4, test_size=0.3, dev_size=0.3):
    assert (train_size + test_size + dev_size == 1.0)
    # Sortiere Dataframe nach den claims, sodass 0 und 1 sich für jeden claim ungefähr die Waage halten sollten.
    # Jeder Claim soll ausschließlich in einem set vorkommen
    samples_sorted = samples.sort_values(by=['text'])

    # Merke immer letzten Eintrag, den man aufgenommen hat, damit sich sets nicht überschneiden
    samples_last_position = 0

    # Gesamtgröße aller Samples
    samples_size = len(samples_sorted)

    # Training Set
    samples_last_position = int(samples_size * train_size)
    train_last_element = samples_sorted.iloc[samples_last_position]["text"]  # Größenangaben der einzelnen Sets werden abgerundet

    while (train_last_element == samples_sorted.iloc[samples_last_position + 1]["text"]):
        samples_last_position += 1
        train_last_element = samples_sorted.iloc[samples_last_position]["text"]

    train = samples_sorted[:samples_last_position + 1]

    train_last_position = samples_last_position

    # Test Set
    samples_last_position = int(train_last_position + samples_size * test_size)

    test_last_element = samples_sorted.iloc[samples_last_position]["text"]  # Größenangaben der einzelnen Sets werden abgerundet

    while (test_last_element == samples_sorted.iloc[samples_last_position + 1]["text"]):
        samples_last_position += 1
        test_last_element = samples_sorted.iloc[samples_last_position]["text"]

    test = samples_sorted.iloc[train_last_position + 1:samples_last_position + 1]

    test_last_position = samples_last_position

    # Dev Set, nimm den rest

    dev = samples_sorted.iloc[test_last_position + 1:]

    print("Train size:")
    train_len = len(train)
    print(train_len)
    print(train_len / samples_size)
    print("Test size:")
    test_len = len(test)
    print(test_len)
    print(test_len / samples_size)
    print("Dev size:")
    dev_len = len(dev)
    print(dev_len)
    print(dev_len / samples_size)

    print(train_len + test_len + dev_len == samples_size)

    print("Last Train/First Test")
    print(train.iloc[-1]["text"])
    print(test.iloc[0]["text"])

    print("Last Test/First Dev")
    print(test.iloc[-1]["text"])
    print(dev.iloc[0]["text"])

    return train, test, dev

if __name__ == '__main__':
    execute()