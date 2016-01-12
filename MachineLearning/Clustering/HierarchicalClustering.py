from LinearUtils.Vectors import distance

def isLeaf(cluster):
    return len(cluster) == 1

def getChildren(cluster):
    """returns the two children of this cluster if it's a merged cluster"""
    if isLeaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]

def getValues(cluster):
    """returns the value in this cluster (if it's a leaf cluster)
    or all the values in the leaf clusters below it (if it's not)"""
    if isLeaf(cluster):
        return cluster
    else:
        return [value
                for child in getChildren(cluster)
                for value in getValues(child)]

def clusterDistance(cluster1, cluster2, distance_agg=min):
    """finds the aggregate distance between elements of cluster1
    and elements of cluster2"""
    return distance_agg([distance(input1, input2)
                         for input1 in getValues(cluster1)
                         for input2 in getValues(cluster2)])

def getMergeOrder(cluster):
    if isLeaf(cluster):
        return float('inf')
    else:
        return cluster[0] # merge_order is first element of 2-tuple

def bottomUpCluster(inputs, distance_agg=min):
    clusters = [(input,) for input in inputs]

    while len(clusters) > 1:
        c1, c2 = min([(cluster1, cluster2)
                     for i, cluster1 in enumerate(clusters)
                     for cluster2 in clusters[:i]],
                     key=lambda p: clusterDistance(p[0], p[1], distance_agg))

        clusters = [c for c in clusters if c != c1 and c != c2]
        mergedCluster = (len(clusters), [c1, c2])
        clusters.append(mergedCluster)

    return clusters[0]

def generateClusters(baseCluster, numClusters):
    clusters = [baseCluster]

    while len(clusters) < numClusters:
        next_cluster = min(clusters, key=getMergeOrder)
        clusters = [c for c in clusters if c != next_cluster]
        clusters.extend(getChildren(next_cluster))

    return clusters

if __name__ == "__main__":
    """
    Hierarchical Clustering Technique
    """
    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    print("Bottom Up Hierarchical Clustering")

    base_cluster = bottomUpCluster(inputs,distance_agg=min)
    print(base_cluster)

    print()
    print("Three clusters, min:")
    for cluster in generateClusters(base_cluster, 3):
        print(getValues(cluster))

    print()
    print("three clusters, max:")
    base_cluster = bottomUpCluster(inputs, distance_agg=max)
    for cluster in generateClusters(base_cluster, 3):
        print(getValues(cluster))
