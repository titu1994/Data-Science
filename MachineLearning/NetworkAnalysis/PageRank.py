import math, random, re
from collections import defaultdict, Counter, deque
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

 # give each user a friends list
for user in users:
    user["friends"] = []

    # and fill it
for i, j in friendships:
    users[i]["friends"].append(users[j]) # add i as a friend of j
    users[j]["friends"].append(users[i]) # add j as a friend of i


endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 3),
                (2, 3), (3, 4), (5, 4), (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

def PageRank(users, damping = 0.85, num_iters = 100):
    """
    A simplified version looks like this:

    1. There is a total of 1.0 (or 100%) PageRank in the network.
    2. Initially this PageRank is equally distributed among nodes.
    3. At each step, a large fraction of each node’s PageRank is distributed evenly among its outgoing links.
    4. At each step, the remainder of each node’s PageRank is distributed evenly among all nodes.

    """

    # initially distribute PageRank evenly
    num_users = len(users)
    pr = { user["id"] : 1 / num_users for user in users }

    # this is the small fraction of PageRank
    # that each node gets each iteration
    base_pr = (1 - damping) / num_users

    for __ in range(num_iters):
        next_pr = { user["id"] : base_pr for user in users }
        for user in users:
            # distribute PageRank to outgoing links
            links_pr = pr[user["id"]] * damping
            for endorsee in user["endorses"]:
                next_pr[endorsee["id"]] += links_pr / len(user["endorses"])

        pr = next_pr

    return pr



if __name__ == "__main__":
    for user in users:
        user["endorses"] = []       # add one list to track outgoing endorsements
        user["endorsed_by"] = []    # and another to track endorsements

    for source_id, target_id in endorsements:
        users[source_id]["endorses"].append(users[target_id])
        users[target_id]["endorsed_by"].append(users[source_id])

    endorsements_by_id = [(user["id"], len(user["endorsed_by"]))
                      for user in users]

    sorted(endorsements_by_id, key=lambda pair: pair[1], reverse=True)

    print("PageRank")
    for user_id, pr in PageRank(users).items():
        print(user_id, pr)






