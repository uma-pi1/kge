import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint

# download link for this checkpoint given under results above
checkpoint = load_checkpoint('wn18-complex.pt')
model = KgeModel.create_from(checkpoint)

#TODO: create kgeembedder to pass values to the model
#TODO: Want to call model.embedder.score_emb() on the embeddings I create
#TODO: how to create these embeddings? Like below, can only find functions to convert index to embedding?

s = torch.Tensor([0, 2,]).long()             # subject indexes
p = torch.Tensor([0, 1,]).long()             # relation indexes
scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects

print(o)
print(model.dataset.entity_strings(s))       # convert indexes to mentions
print(model.dataset.relation_strings(p))
print(model.dataset.entity_strings(o))

#TODO: first: does complex create embeddings or just use them??? if just use then much simpler I think but I don't understyand how that works in that case
# Output (slightly revised for readability):
#
# tensor([8399, 8855])
# ['Dominican Republic'        'Mighty Morphin Power Rangers']
# ['has form of government'    'is tv show with actor']
# ['Republic'                  'Johnny Yong Bosch']

s = model.dataset.index('Dominican Republic')
print(s)
