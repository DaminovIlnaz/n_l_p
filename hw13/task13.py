import numpy as np
import tabulate as tb


def edit_distance_dp(first_word, second_word):
    n = len(first_word) + 1
    m = len(second_word) + 1

    # initialize D matrix
    D = np.zeros(shape=(n, m), dtype=np.int)
    D[:, 0] = range(n)
    D[0, :] = range(m)

    B = np.zeros(shape=(n, m), dtype=[("del", 'b'),
                                      ("sub", 'b'),
                                      ("ins", 'b')])
    B[1:, 0] = (1, 0, 0)
    B[0, 1:] = (0, 0, 1)

    for i, l_1 in enumerate(first_word, start=1):
        for j, l_2 in enumerate(second_word, start=1):
            deletion = D[i - 1, j] + 1
            insertion = D[i, j - 1] + 1
            substitution = D[i - 1, j - 1] + (0 if l_1 == l_2 else 2)
            mo = np.min([deletion, insertion, substitution])
            B[i, j] = (deletion == mo, substitution == mo, insertion == mo)
            D[i, j] = mo
    return D, B


def backtrace(b_matrix):
    i, j = b_matrix.shape[0] - 1, b_matrix.shape[1] - 1
    backtrace_ids = [(i, j)]

    while (i, j) != (0, 0):
        if b_matrix[i, j][1]:
            i, j = i - 1, j - 1
        elif b_matrix[i, j][0]:
            i, j = i - 1, j
        elif b_matrix[i, j][2]:
            i, j = i, j - 1
        backtrace_ids.append((i, j))

    return backtrace_ids


def alignment(first_word, second_word, backtrace):
    aligned_word_1 = []
    aligned_word_2 = []
    operations = []

    backtrace = backtrace[::-1]

    for k in range(len(backtrace) - 1):
        i_0, j_0 = backtrace[k]
        i_1, j_1 = backtrace[k + 1]

        w_1_letter = None
        w_2_letter = None
        op = None

        if i_1 > i_0 and j_1 > j_0:
            if first_word[i_0] == second_word[j_0]:
                w_1_letter = first_word[i_0]
                w_2_letter = second_word[j_0]
                op = " "
            else:
                w_1_letter = first_word[i_0]
                w_2_letter = second_word[j_0]
                op = "s"
        elif i_0 == i_1:  # insertion
            w_1_letter = " "
            w_2_letter = second_word[j_0]
            op = "i"
        else:  # j_0 == j_1,  deletion
            w_1_letter = first_word[i_0]
            w_2_letter = " "
            op = "d"

        aligned_word_1.append(w_1_letter)
        aligned_word_2.append(w_2_letter)
        operations.append(op)

    return aligned_word_1, aligned_word_2, operations


def reverse_table(table, height):
    result = []
    n = height
    while n > 0:
        n -= 1
        result.append(table[n])
    return result


def make_table(first_word, second_word, D, B, backtrace):
    w_1 = first_word.upper()
    w_2 = second_word.upper()

    w_1 = "#" + w_1
    w_2 = "#" + w_2

    table = []

    table.append([""] + list(w_2))
    max_n_len = len(str(np.max(D)))

    for i, l_1 in enumerate(w_1):
        row = [l_1]

        for j, l_2 in enumerate(w_2):
            v, d, h = B[i, j]
            direction = ("down." if v else "") + \
                        ("down-left." if d else "") + \
                        ("left." if h else "")

            # ↓ ⇙

            dist = str(D[i, j])

            cell_str = "{direction} {star}{dist}{star}".format(
                direction=direction,
                star=" *"[((i, j) in backtrace)],
                dist=dist)
            row.append(cell_str)
        table.append(row)

    table = reverse_table(table, len(w_1))

    return table


fist_word = "intention"
second_word = "execution"

D, B = edit_distance_dp(fist_word, second_word)
bt = backtrace(B)
print(D)

edit_distance_table = make_table(fist_word, second_word, D, B, bt)
alignment_table = alignment(fist_word, second_word, bt)

print("Minimum edit distance with backtrace:")
print(tb.tabulate(edit_distance_table, stralign="right", tablefmt="orgtbl"))

print("\nAlignment:")
print(tb.tabulate(alignment_table, tablefmt="orgtbl"))