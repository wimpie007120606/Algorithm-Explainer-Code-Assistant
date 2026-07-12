# Tree Data Structures

## Binary Search Tree (BST)

A Binary Search Tree is a node-based binary tree where for every node:
- All values in the left subtree are less than the node's value.
- All values in the right subtree are greater than the node's value.

**Operations and Complexity:**
- Search: O(h) where h is the height. O(log n) average, O(n) worst case (degenerate/skewed tree).
- Insert: O(h) average O(log n), worst O(n).
- Delete: O(h).

### Java Implementation — BST Insert and Search

```java
public class BST {
    static class Node {
        int val;
        Node left, right;
        Node(int val) { this.val = val; }
    }

    public Node insert(Node root, int val) {
        if (root == null) return new Node(val);
        if (val < root.val) root.left = insert(root.left, val);
        else if (val > root.val) root.right = insert(root.right, val);
        return root;
    }

    public boolean search(Node root, int val) {
        if (root == null) return false;
        if (val == root.val) return true;
        return val < root.val ? search(root.left, val) : search(root.right, val);
    }
}
```

## AVL Tree

An AVL Tree is a self-balancing Binary Search Tree where the height difference (balance factor) between left and right subtrees of any node is at most 1. Named after Adelson-Velsky and Landis.

**Balance factor** = height(left subtree) - height(right subtree). Valid values: -1, 0, +1.

**Time Complexity:**
- Search, Insert, Delete: O(log n) guaranteed.

**Rotations** keep the tree balanced:
- Left Rotation (LL case): applied when right subtree is heavy.
- Right Rotation (RR case): applied when left subtree is heavy.
- Left-Right Rotation (LR case): left rotate child, then right rotate root.
- Right-Left Rotation (RL case): right rotate child, then left rotate root.

### Java Implementation — AVL Insert with Rotation

```java
public class AVLTree {
    static class Node {
        int val, height;
        Node left, right;
        Node(int val) { this.val = val; this.height = 1; }
    }

    private int height(Node n) { return n == null ? 0 : n.height; }
    private int getBalance(Node n) { return n == null ? 0 : height(n.left) - height(n.right); }

    private Node rightRotate(Node y) {
        Node x = y.left, T2 = x.right;
        x.right = y; y.left = T2;
        y.height = Math.max(height(y.left), height(y.right)) + 1;
        x.height = Math.max(height(x.left), height(x.right)) + 1;
        return x;
    }

    private Node leftRotate(Node x) {
        Node y = x.right, T2 = y.left;
        y.left = x; x.right = T2;
        x.height = Math.max(height(x.left), height(x.right)) + 1;
        y.height = Math.max(height(y.left), height(y.right)) + 1;
        return y;
    }

    public Node insert(Node node, int val) {
        if (node == null) return new Node(val);
        if (val < node.val) node.left = insert(node.left, val);
        else if (val > node.val) node.right = insert(node.right, val);
        else return node;

        node.height = 1 + Math.max(height(node.left), height(node.right));
        int balance = getBalance(node);

        if (balance > 1 && val < node.left.val) return rightRotate(node);      // LL
        if (balance < -1 && val > node.right.val) return leftRotate(node);     // RR
        if (balance > 1 && val > node.left.val) {                              // LR
            node.left = leftRotate(node.left);
            return rightRotate(node);
        }
        if (balance < -1 && val < node.right.val) {                            // RL
            node.right = rightRotate(node.right);
            return leftRotate(node);
        }
        return node;
    }
}
```

## Red-Black Tree

A Red-Black Tree is a self-balancing BST where each node has a color (red or black), satisfying:
1. Every node is red or black.
2. The root is always black.
3. Red nodes cannot have red children (no two consecutive red nodes).
4. Every path from any node to its null descendants has the same number of black nodes (black-height property).

**Time Complexity:** O(log n) for search, insert, and delete — guaranteed.

Red-Black Trees are used in Java's `TreeMap` and `TreeSet`, and in Linux kernel's `CFS` scheduler.

## Segment Tree

A Segment Tree stores intervals or segments, allowing efficient range queries and updates.

**Time Complexity:**
- Build: O(n)
- Query: O(log n)
- Update: O(log n)

**Space Complexity:** O(4n) for array-based implementation.

### Java Implementation — Segment Tree (Range Sum)

```java
public class SegmentTree {
    private int[] tree;
    private int n;

    public SegmentTree(int[] arr) {
        n = arr.length;
        tree = new int[4 * n];
        build(arr, 0, 0, n - 1);
    }

    private void build(int[] arr, int node, int start, int end) {
        if (start == end) { tree[node] = arr[start]; return; }
        int mid = (start + end) / 2;
        build(arr, 2 * node + 1, start, mid);
        build(arr, 2 * node + 2, mid + 1, end);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }

    public int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;
        if (l <= start && end <= r) return tree[node];
        int mid = (start + end) / 2;
        return query(2 * node + 1, start, mid, l, r)
             + query(2 * node + 2, mid + 1, end, l, r);
    }

    public void update(int node, int start, int end, int idx, int val) {
        if (start == end) { tree[node] = val; return; }
        int mid = (start + end) / 2;
        if (idx <= mid) update(2 * node + 1, start, mid, idx, val);
        else update(2 * node + 2, mid + 1, end, idx, val);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
}
```
