# Graph Algorithms

## Breadth-First Search (BFS)

BFS explores a graph level by level using a queue data structure.
Starting from a source vertex, it visits all adjacent vertices before moving to the next level.

**Time Complexity:** O(V + E) where V is vertices and E is edges.
**Space Complexity:** O(V) for the queue and visited array.

### Java Implementation

```java
import java.util.*;

public class BFS {
    public void bfs(int start, List<List<Integer>> adj) {
        boolean[] visited = new boolean[adj.size()];
        Queue<Integer> queue = new LinkedList<>();
        visited[start] = true;
        queue.add(start);

        while (!queue.isEmpty()) {
            int v = queue.poll();
            System.out.print(v + " ");
            for (int u : adj.get(v)) {
                if (!visited[u]) {
                    visited[u] = true;
                    queue.add(u);
                }
            }
        }
    }
}
```

## Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking.
It uses a stack (implicit via recursion or explicit).

**Time Complexity:** O(V + E)
**Space Complexity:** O(V) for the recursion stack.

## Ford-Fulkerson Algorithm

The Ford-Fulkerson algorithm computes the maximum flow in a flow network.
It works by repeatedly finding augmenting paths from source to sink and
increasing the flow along those paths until no augmenting path exists.

### Max-Flow Min-Cut Theorem

The maximum flow value from source s to sink t equals the minimum capacity
of any s-t cut in the network. This is the fundamental theorem of network flows.

## Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path from a source vertex to all
other vertices in a weighted graph with non-negative edge weights.

**Time Complexity:** O((V + E) log V) with a binary heap.
**Key requirement:** All edge weights must be non-negative.

```java
import java.util.*;

public class Dijkstra {
    public int[] dijkstra(int src, int[][] graph) {
        int n = graph.length;
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        pq.offer(new int[]{0, src});

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int d = curr[0], u = curr[1];
            if (d > dist[u]) continue;
            for (int v = 0; v < n; v++) {
                if (graph[u][v] != 0 && dist[u] + graph[u][v] < dist[v]) {
                    dist[v] = dist[u] + graph[u][v];
                    pq.offer(new int[]{dist[v], v});
                }
            }
        }
        return dist;
    }
}
```

# Data Structures

## Quadtree

A quadtree is a tree data structure where each internal node has exactly four children,
used to partition two-dimensional space by recursively subdividing it into four quadrants.

### Insertion

To insert a point into a quadtree:
1. If the current node is a leaf and has capacity, store the point here.
2. If the node is at capacity, subdivide into four quadrants (NE, NW, SE, SW).
3. Re-insert existing points into the appropriate quadrant.
4. Insert the new point into the appropriate quadrant.

**Time Complexity:** O(log n) average for uniformly distributed points.

