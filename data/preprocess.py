import argparse


def index(symbols, file):
  with open(file, 'w') as f:
    for i, k in symbols.items():
      f.write(str(k) + '\t' + str(i) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--folder', type=str)
  args = parser.parse_args()

  print('Preprocessing ' + args.folder)
  raw_split_files = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
  split_files = {'train': 'train.del', 'valid': 'valid.del', 'test': 'test.del'}

  # read data and collect entity and relation names
  raw = {}
  entities = {}
  relations = {}
  ent_id = 0
  rel_id = 0
  for k, file in raw_split_files.items():
      with open(args.folder + '/' + file, 'r') as f:
          raw[k] = list(map(lambda s: s.strip().split('\t'), f.readlines()))
          for t in raw[k]:
              if t[0] not in entities:
                  entities[t[0]] = ent_id
                  ent_id += 1
              if t[1] not in relations:
                  relations[t[1]] = rel_id
                  rel_id += 1
              if t[2] not in entities:
                  entities[t[2]] = ent_id
                  ent_id += 1
          print(str(len(raw[k])) + ' triples in ' + file)

  print(str(len(relations)) + ' distinct relations')
  print(str(len(entities)) + ' distinct entities')
  print('Writing indexes...')
  index(relations, args.folder + "/relation_map.del")
  index(entities, args.folder + "/entity_map.del")

  # write out
  print('Writing triples...')
  for k, file in split_files.items():
      with open(args.folder + '/' + file, 'w') as f:
          for t in raw[k]:
              f.write(str(entities[t[0]]))
              f.write('\t')
              f.write(str(relations[t[1]]))
              f.write('\t')
              f.write(str(entities[t[2]]))
              f.write('\n')
  print('Done')
