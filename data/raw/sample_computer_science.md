# Computer Science Study Notes

## Algorithm Analysis

Big-O notation describes how runtime or memory grows as input size increases.
It focuses on dominant growth rather than exact machine time.

Common growth rates:

- O(1): constant time
- O(log n): logarithmic time
- O(n): linear time
- O(n log n): efficient comparison sorting
- O(n^2): nested pairwise work
- O(2^n): exponential search over subsets

## Data Structures

Arrays provide fast indexed access. Linked lists make insertion cheap when the
node location is already known. Hash tables provide average O(1) lookup when the
hash function distributes keys well.

Stacks are last-in, first-out. Queues are first-in, first-out. Priority queues
remove the item with highest or lowest priority.

## Graphs

Graphs model relationships between objects. Vertices are entities and edges are
connections.

Breadth-first search finds shortest paths in unweighted graphs. Depth-first
search is useful for traversal, cycle detection, and topological sorting.

Dijkstra's algorithm finds shortest paths with non-negative weights. Bellman-Ford
handles negative edges and can detect negative cycles.

## Dynamic Programming

Dynamic programming solves problems by combining answers to overlapping
subproblems. A useful checklist:

1. Define the state.
2. Write the recurrence.
3. Identify base cases.
4. Choose top-down memoization or bottom-up tabulation.
5. Recover the final answer from the state table.

Classic examples include Fibonacci numbers, longest common subsequence, knapsack,
edit distance, and shortest paths in directed acyclic graphs.

