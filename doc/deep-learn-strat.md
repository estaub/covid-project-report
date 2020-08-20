Topic classification strategy

### Target

A set of target topics will be hand-identified in the N most popular/active github repo records.
Correct labelling may use "hidden" knowledge that is not in a record; labelling represents the
truth of the repo, not just the best that might be deduced from the records.

These are recorded as JSON of form:
```json
[
    { "url": "x.github.com/foo/proj", "topics": [ "astronomy", "dating"]}
]
```

These may be merged or updated with additional target records.

# Preprocessing

The preprocessing of input records is just BertTokenizer.  
They are also split into subdocuments of 510 tokens or less.  
These may be split on sentence boundaries and/or made to overlap.
It probably makes sense to cache the data at this point.

# Input updates

Github updates weekly.  Training must be coherent across updates.  

# Loss function

This begs the question of how many topics to attempt to classify for in a single model.  It may make sense to make a separate model for each, but it may be much slower. 

The cost of missing a topic that is in hundreds of other repos is unimportant, but a topic that is unique to one repo "must" be detected.
The cost of a false positive is unclear.

It's important to capture the losses for each topic individually, so that we can see if some are doing well.

# Model design

https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforsequenceclassification#transformers.BertForSequenceClassification

https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5

https://www.youtube.com/watch?v=_eSGWNqKeeY&t=15s
?

