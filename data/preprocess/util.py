
def store_map(symbol_map, filename):
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")


def process_raw_split_files(raw_split_files: dict, folder: str,  order_sop: bool = False):

    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # read data and collect entities and relations
    split_sizes = {}
    raw = {}
    entities = {}
    relations = {}
    entities_in_train = {}
    relations_in_train = {}
    ent_id = 0
    rel_id = 0
    for split, filename in raw_split_files.items():
        with open(folder + "/" + filename, "r") as f:
            raw[split] = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            for t in raw[split]:
                if t[S] not in entities:
                    entities[t[S]] = ent_id
                    ent_id += 1
                if t[P] not in relations:
                    relations[t[P]] = rel_id
                    rel_id += 1
                if t[O] not in entities:
                    entities[t[O]] = ent_id
                    ent_id += 1
            print(
                f"Found {len(raw[split])} triples in {split} split "
                f"(file: {filename})."
            )
            split_sizes[split] = len(raw[split])
            if "train" in split:
                entities_in_train = entities.copy()
                relations_in_train = relations.copy()

    return split_sizes, raw,  entities, relations, entities_in_train, relations_in_train
