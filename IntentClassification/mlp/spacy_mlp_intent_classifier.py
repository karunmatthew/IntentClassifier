import spacy

# this contains a meta.json, that lists the pipelines that the doc creation
# takes. the pipeline can be viewed with nlp.pipe_names or nlp.pipelines
nlp = spacy.load("en_core_web_lg")

# batch the conversion to doc object
# this is faster
batch_data = ['Step to the left', 'turn right after walking forward two steps']
docs = list(nlp.pipe(batch_data))


# you can pass in data and context as tuples
data = [
    ('Step to the left', {'intent': 'GoToLocation', 'agent_pos': []}),
    ('turn right after walking forward two steps', {'intent': 'GoToLocation',
                                                    'agent_pos': []})
]

for doc, context in nlp.pipe(data, as_tuples=True):
    print(doc.text, context['intent'])
    # you can even add context to extended custom properties of doc
    

