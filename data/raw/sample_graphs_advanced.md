# Advanced Graph Algorithms

## Bellman-Ford Algorithm

Bellman-Ford finds shortest paths from a single source to all vertices in a weighted directed graph, including graphs with negative-weight edges. Unlike Dijkstra, it can handle negative weights but not negative cycles.

**Algorithm:**
1. Initialize all distances to infinity except the source (distance 0).
2. Relax all edges |V| - 1 times.
3. Check for negative cycles: if any edge can still be relaxed, a negative cycle exists.

**Time Complexity:** O(V × E)
**Space Complexity:** O(V)

**Key difference from Dijkstra:** Bellman-Ford works with negative edge weights. Dijkstra requires non-negative weights.

### Java Implementation — Bellman-Ford

```java
public class BellmanFord {
    static class Edge {
        int src, dst, weight;
        Edge(int s, int d, int w) { src = s; dst = d; weight = w; }
    }

    public int[] bellmanFord(int V, List<Edge> edges, int src) {
        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;

        for (int i = 0; i < V - 1; i++) {
            for (Edge e : edges) {
                if (dist[e.src] != Integer.MAX_VALUE &&
                    dist[e.src] + e.weight < dist[e.dst]) {
                    dist[e.dst] = dist[e.src] + e.weight;
                }
            }
        }

        // Negative cycle detection
        for (Edge e : edges) {
            if (dist[e.src] != Integer.MAX_VALUE &&
                dist[e.src] + e.weight < dist[e.dst]) {
                throw new RuntimeException("Graph contains a negative cycle");
            }
        }
        return dist;
    }
}
```

## Topological Sort

Topological Sort orders vertices of a Directed Acyclic Graph (DAG) so that for every directed edge u → v, vertex u appears before v.

**Applications:** Task scheduling, build systems (Maven, Gradle), course prerequisite ordering.

**Two approaches:**
1. Kahn's Algorithm (BFS-based, uses in-degree array)
2. DFS-based (uses a stack, records finish times)

**Time Complexity:** O(V + E)

### Java Implementation — Kahn's Algorithm

```java
public class TopologicalSort {
    public List<Integer> topoSort(int V, List<List<Integer>> adj) {
        int[] inDegree = new int[V];
        for (int u = 0; u < V; u++)
            for (int v : adj.get(u)) inDegree[v]++;

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < V; i++)
            if (inDegree[i] == 0) queue.add(i);

        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            result.add(u);
            for (int v : adj.get(u)) {
                if (--inDegree[v] == 0) queue.add(v);
            }
        }

        if (result.size() != V) throw new RuntimeException("Graph has a cycle — not a DAG");
        return result;
    }
}
```

## Prim's Algorithm (Minimum Spanning Tree)

Prim's algorithm builds a Minimum Spanning Tree (MST) by greedily adding the cheapest edge that connects the growing tree to a new vertex.

**Time Complexity:** O((V + E) log V) with a priority queue.

### Java Implementation — Prim's MST

```java
public class Prims {
    public int primMST(int V, int[][] graph) {
        int[] key = new int[V];
        boolean[] inMST = new boolean[V];
        Arrays.fill(key, Integer.MAX_VALUE);
        key[0] = 0;
        int totalWeight = 0;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        pq.offer(new int[]{0, 0});

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int u = curr[1];
            if (inMST[u]) continue;
            inMST[u] = true;
            totalWeight += curr[0];
            for (int v = 0; v < V; v++) {
                if (graph[u][v] != 0 && !inMST[v] && graph[u][v] < key[v]) {
                    key[v] = graph[u][v];
                    pq.offer(new int[]{key[v], v});
                }
            }
        }
        return totalWeight;
    }
}
```

## Kruskal's Algorithm (Minimum Spanning Tree)

Kruskal's builds an MST by sorting all edges by weight and greedily adding edges that do not form a cycle (using Union-Find).

**Time Complexity:** O(E log E) — dominated by edge sorting.

### Java Implementation — Kruskal's with Union-Find

```java
public class Kruskals {
    private int[] parent, rank;

    public int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    public boolean union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) { int t = px; px = py; py = t; }
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    }

    public int kruskalMST(int V, int[][] edges) {
        parent = new int[V]; rank = new int[V];
        for (int i = 0; i < V; i++) parent[i] = i;
        Arrays.sort(edges, Comparator.comparingInt(e -> e[2]));
        int total = 0;
        for (int[] e : edges) {
            if (union(e[0], e[1])) total += e[2];
        }
        return total;
    }
}
```

## Floyd-Warshall Algorithm (All-Pairs Shortest Paths)

Floyd-Warshall computes shortest paths between ALL pairs of vertices.

**Recurrence:** `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])` for each intermediate vertex k.

**Time Complexity:** O(V³)
**Space Complexity:** O(V²)

**Key property:** Works with negative edge weights (not negative cycles). Detects negative cycles if `dist[i][i] < 0` after the algorithm.

### Java Implementation — Floyd-Warshall

```java
public class FloydWarshall {
    public int[][] floydWarshall(int V, int[][] graph) {
        int[][] dist = new int[V][V];
        for (int[] row : dist) Arrays.fill(row, Integer.MAX_VALUE / 2);
        for (int i = 0; i < V; i++) {
            dist[i][i] = 0;
            for (int j = 0; j < V; j++)
                if (graph[i][j] != 0) dist[i][j] = graph[i][j];
        }
        for (int k = 0; k < V; k++)
            for (int i = 0; i < V; i++)
                for (int j = 0; j < V; j++)
                    dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
        return dist;
    }
}
```
