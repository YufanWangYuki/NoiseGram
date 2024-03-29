import random
import errant
import pdb
random.seed(1)

def get_sentences(data_path, num=-1):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    if num > 0:
        print("Here Type ", type(num))
        random.shuffle(lines)
        lines = lines[:num]
    # pdb.set_trace()
    # if 'fce' in data_path:
    #     texts = [' '.join(l.rstrip('\n').split()[:]) for l in lines]
    #     ids = [str(id) for id in range(len(lines))]
    # else:
    texts = []
    ids = []
    for idx, l in enumerate(lines):
        if len(l.rstrip('\n').split()) < 1:
            print(idx)
            continue
        texts.append(' '.join(l.rstrip('\n').split()[1:]))
        ids.append(l.rstrip('\n').split()[0])

    # Remove space before full stops at end
    texts = [t[:-2]+'.' if t[-2:]==' .' else t for t in texts]
    # pdb.set_trace()
    return ids, texts

def correct(model, sentence):

    '''Gramformer decoding'''

    result = model.correct(sentence, max_candidates=1)[0]
    return result

def count_edits(input, prediction):
    '''
    Count number of edits
    '''
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    return len(edits)

def return_edits(input, prediction):
    '''
    Get edits
    '''
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    for e in edits:
        e = annotator.classify(e)
    return edits