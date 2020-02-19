from itertools import combinations, permutations
from partition_utils import Partition, get_mentions
from typing import Optional, TypeVar, Tuple

T = TypeVar('T')


def get_entity(partition: Partition, mention: T) -> Optional[T]:
    for entity in partition:
        if mention in entity:
            return entity


def lea_sub(keys: Partition, response: Partition) -> float:
    correct_link_ratio = 0
    nb_mentions = 0

    for key_entity in keys:
        key_entity_size = len(key_entity)
        nb_correct_links = 0
        if key_entity_size == 1:
            c_mention = [i for i in key_entity][0]
            r_entity = get_entity(response, c_mention)
            if r_entity is not None and len(r_entity) == 1:
                nb_correct_links += 1
        else:
            for c_mention, n_mention in combinations(key_entity, 2):
                if get_entity(response, c_mention) is not None:
                    if get_entity(response, c_mention) == get_entity(response, n_mention):
                        nb_correct_links += 1

        nb_entity_links = 1
        if key_entity_size != 1:
            nb_entity_links = (key_entity_size * (key_entity_size - 1)) / 2

        correct_link_ratio += key_entity_size * (nb_correct_links / nb_entity_links)
        nb_mentions += key_entity_size
    return correct_link_ratio / nb_mentions


def lea(keys: Partition, response: Partition) -> Tuple[float, float, float]:
    R = lea_sub(keys, response)
    P = lea_sub(response, keys)
    F = (2 * P * R) / (P + R) if R+P != 0 else 0
    return R, P, F


def edit(keys: Partition, response: Partition) -> float:
    return max((sum(len(a.intersection(b)) for a, b in zip(keys, perm)) for perm in permutations(response))) / len(get_mentions(keys))

def edit2(keys, response):
    {}
