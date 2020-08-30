import numpy as np
import sys

def main(args):
    chicken = (np.random.random([10]) * 4).astype(np.uint8)
    turkey = (np.random.random([10]) * 4).astype(np.uint8)

    comparison, chicken_subsequences, turkey_subsequences = compare_subsequences(chicken, turkey, 5)

    match_probability = comparison.mean(axis=-1)

    best_match = np.unravel_index(match_probability.argmax(), match_probability.shape)

    highest_probability = match_probability[best_match]
    best_chicken_subsequence = chicken_subsequences[best_match[0]]
    best_turkey_subsequence = turkey_subsequences[best_match[1]]

    print('best chicken subsequence')
    pretty_print(best_chicken_subsequence)

    print('best turkey subsequence')
    pretty_print(best_turkey_subsequence)

    print(f'match percentage ${highest_probability}')


def get_subsequences(dna, subsequence_length):
    sub_sequence_count = dna.shape[-1] - subsequence_length

    sub_sequence_start_indexes = np.arange(sub_sequence_count)
    sub_sequence_relative_indexes = np.arange(subsequence_length)
    all_indexes = sub_sequence_start_indexes[:, np.newaxis] + sub_sequence_relative_indexes[np.newaxis, :]
    subsequences = dna[all_indexes]

    return subsequences

def compare_subsequences(chicken, turkey, subsequence_length):
    chicken_subsequences = get_subsequences(chicken, subsequence_length)[:, np.newaxis, :]
    turkey_subsequences = get_subsequences(turkey, subsequence_length)[np.newaxis, :, :]

    match = (chicken_subsequences == turkey_subsequences)

    return match, chicken_subsequences[:, 0, :], turkey_subsequences[0, :, :]

def pretty_print(dna):
    to_nucleotide_letter = np.array(list('actg'))
    pretty_dna = to_nucleotide_letter[dna]

    print(''.join(pretty_dna.tolist()))


def some_simple_sequence_comparison():
    SEQUENCE_COUNT = 3
    SEQUENCE_LENGTH = 30

    chicken_dna = (np.random.random([SEQUENCE_COUNT, SEQUENCE_LENGTH]) * 4).astype(np.uint8)
    dino_dna = (np.random.random([SEQUENCE_COUNT, SEQUENCE_LENGTH]) * 4).astype(np.uint8)

    match = (chicken_dna == dino_dna)

    for i in range(SEQUENCE_COUNT):
        print(chicken_dna[i, :])
        print(dino_dna[i, :])

        print(match[i, :].astype(np.int8))

        print("\n")

    print(match.mean())

    print(match.mean(axis=1))




if __name__ == '__main__':
    main(sys.argv)