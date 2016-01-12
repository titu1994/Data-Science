from collections import deque
from LinearUtils.Vectors import dotProduct, magnitude, scalarMultiply, shape, distance
from LinearUtils.Matrices import  getRow, getCol, generateMatrix
from functools import partial

# Code from Data Science from Scratch - github

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#
# Betweenness Centrality
#

def shortest_paths_from(from_user):

    # a dictionary from "user_id" to *all* shortest paths to that user
    shortest_paths_to = { from_user["id"] : [[]] }

    # a queue of (previous user, next user) that we need to check.
    # starts out with all pairs (from_user, friend_of_from_user)
    frontier = deque((from_user, friend)
                     for friend in from_user["friends"])

    # keep going until we empty the queue
    while frontier:

        prev_user, user = frontier.popleft() # take from the beginning
        user_id = user["id"]

        # the fact that we're pulling from our queue means that
        # necessarily we already know a shortest path to prev_user
        paths_to_prev = shortest_paths_to[prev_user["id"]]
        paths_via_prev = [path + [user_id] for path in paths_to_prev]

        # it's possible we already know a shortest path to here as well
        old_paths_to_here = shortest_paths_to.get(user_id, [])

        # what's the shortest path to here that we've seen so far?
        if old_paths_to_here:
            min_path_length = len(old_paths_to_here[0])
        else:
            min_path_length = float('inf')

        # any new paths to here that aren't too long
        new_paths_to_here = [path_via_prev
                             for path_via_prev in paths_via_prev
                             if len(path_via_prev) <= min_path_length
                             and path_via_prev not in old_paths_to_here]

        shortest_paths_to[user_id] = old_paths_to_here + new_paths_to_here

        # add new neighbors to the frontier
        frontier.extend((user, friend)
                        for friend in user["friends"]
                        if friend["id"] not in shortest_paths_to)

    return shortest_paths_to

"""
Closeness Centrality
"""

def farness(user):
    """the sum of the lengths of the shortest paths to each other user"""
    return sum(len(paths[0])
               for paths in user["shortest_paths"].values())

"""
Eigenvector Centrality
"""

def matrix_product_entry(A, B, i, j):
    return dotProduct(getRow(A, i), getCol(B, j))

def matrix_multiply(A, B):
    n1, k1 = shape(A)
    n2, k2 = shape(B)
    if k1 != n2:
        raise ArithmeticError("incompatible shapes!")

    return generateMatrix(n1, k2, partial(matrix_product_entry, A, B))

def vector_as_matrix(v):
    """returns the vector v (represented as a list) as a n x 1 matrix"""
    return [[v_i] for v_i in v]

def vector_from_matrix(v_as_matrix):
    """returns the n x 1 matrix as a list of values"""
    return [row[0] for row in v_as_matrix]

def matrix_operate(A, v):
    v_as_matrix = vector_as_matrix(v)
    product = matrix_multiply(A, v_as_matrix)
    return vector_from_matrix(product)

def find_eigenvector(A, tolerance=0.00001):
    guess = [1 for __ in A]

    while True:
        result = matrix_operate(A, guess)
        length = magnitude(result)
        next_guess = scalarMultiply(1/length, result)

        if distance(guess, next_guess) < tolerance:
            return next_guess, length # eigenvector, eigenvalue

        guess = next_guess

def entry_fn(i, j):
    return 1 if (i, j) in friendships or (j, i) in friendships else 0



if __name__ == "__main__":
    """
    Betweenness Centrality
    """

    print("Betweenness Centrality")

    # give each user a friends list
    for user in users:
        user["friends"] = []

    # and fill it
    for i, j in friendships:
        users[i]["friends"].append(users[j]) # add i as a friend of j
        users[j]["friends"].append(users[i]) # add j as a friend of i

    for user in users:
        user["shortest_paths"] = shortest_paths_from(user)


    for user in users:
        user["betweenness_centrality"] = 0.0

    for source in users:
        source_id = source["id"]
        for target_id, paths in source["shortest_paths"].items():
            if source_id < target_id:   # don't double count
                num_paths = len(paths)  # how many shortest paths?
                contrib = 1 / num_paths # contribution to centrality
                for path in paths:
                    for id in path:
                        if id not in [source_id, target_id]:
                            users[id]["betweenness_centrality"] += contrib

    for user in users:
        print("User ID : ", user["id"], " Betweenness Centrality : ", user["betweenness_centrality"])
    print()

    """
    Closeness Centrality
    """

    for user in users:
        user["closeness_centrality"] = 1 / farness(user)

    print("Closeness Centrality")
    for user in users:
        print("User ID : ", user["id"], " Closeness Centrality : ", user["closeness_centrality"])
    print()


    """
    Eigenvector Centralities,
    """
    n = len(users)
    adjacency_matrix = generateMatrix(n, n, entry_fn)

    eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)

    print("Eigenvector Centrality")
    for user_id, centrality in enumerate(eigenvector_centralities):
        print("User ID : ", user_id, " Eigenvector Centrality : ", centrality)
    print()