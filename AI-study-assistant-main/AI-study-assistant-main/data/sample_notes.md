# Study Notes — Binary Search Trees & Graph Traversal

## Binary Search Tree (BST)

A Binary Search Tree is a binary tree where, for every node:
- All values in the left subtree are smaller than the node's value.
- All values in the right subtree are larger than the node's value.

**Average case complexity:** O(log n) for search, insert, and delete, because each
comparison eliminates roughly half of the remaining nodes.

**Worst case complexity:** O(n), which happens when the tree becomes skewed —
for example, inserting sorted data 1, 2, 3, 4, 5 in order creates a straight line, not a tree.

This is exactly why AVL trees exist: they add automatic rotations to keep the
tree balanced, guaranteeing O(log n) even in the worst case.

## AVL Tree Rotations

There are four rotation cases, based on where the imbalance occurs:
- **LL (Left-Left):** single right rotation.
- **RR (Right-Right):** single left rotation.
- **LR (Left-Right):** left rotation on the child, then right rotation on the node.
- **RL (Right-Left):** right rotation on the child, then left rotation on the node.

## Graph Traversal: BFS vs DFS

**Breadth-First Search (BFS)**
- Explores neighbors level by level.
- Uses a queue (FIFO).
- Finds the shortest path in an unweighted graph.
- Time complexity: O(V + E).

**Depth-First Search (DFS)**
- Explores as far as possible along a branch before backtracking.
- Uses a stack, or recursion (which uses the call stack implicitly).
- Useful for cycle detection, topological sorting, and finding connected components.
- Time complexity: O(V + E).

## Quick Comparison Table

| Feature | BST | AVL Tree |
|---|---|---|
| Balancing | None | Automatic via rotations |
| Worst-case search | O(n) | O(log n) |
| Insert complexity | O(n) worst case | O(log n) guaranteed |
| Best use case | Simple, infrequent lookups | Frequent inserts/deletes needing guaranteed speed |
