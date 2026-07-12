# Dynamic Programming

## What is Dynamic Programming?

Dynamic Programming (DP) is an algorithmic technique that solves complex problems by breaking them into overlapping subproblems and storing their solutions to avoid redundant computation.

**Two approaches:**
- **Top-down (Memoization):** Recursive with caching of results.
- **Bottom-up (Tabulation):** Iterative, filling a table from smallest subproblems.

DP applies when a problem has:
1. **Optimal Substructure:** Optimal solution contains optimal solutions to subproblems.
2. **Overlapping Subproblems:** Same subproblems recur multiple times.

## Longest Common Subsequence (LCS)

Given two strings, find the length of their longest common subsequence (not necessarily contiguous).

**Recurrence:**
- If `s1[i] == s2[j]`: `dp[i][j] = dp[i-1][j-1] + 1`
- Else: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

**Time Complexity:** O(m × n) where m, n are string lengths.
**Space Complexity:** O(m × n) for the DP table; reducible to O(min(m,n)) with space optimization.

### Java Implementation — LCS

```java
public class LCS {
    public int lcs(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1))
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                else
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }
}
```

## 0/1 Knapsack Problem

Given n items each with a weight and value, and a knapsack capacity W, find the maximum value subset where total weight does not exceed W. Each item can be taken at most once.

**Recurrence:**
- `dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])` if `weight[i] <= w`
- `dp[i][w] = dp[i-1][w]` otherwise

**Time Complexity:** O(n × W)
**Space Complexity:** O(n × W); reducible to O(W) with 1D array.

### Java Implementation — 0/1 Knapsack

```java
public class Knapsack {
    public int knapsack(int W, int[] weights, int[] values, int n) {
        int[][] dp = new int[n + 1][W + 1];
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= W; w++) {
                dp[i][w] = dp[i - 1][w];
                if (weights[i - 1] <= w)
                    dp[i][w] = Math.max(dp[i][w],
                                        dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            }
        }
        return dp[n][W];
    }
}
```

## Longest Increasing Subsequence (LIS)

Find the length of the longest strictly increasing subsequence of an array.

**O(n²) DP solution:**
- `dp[i]` = length of LIS ending at index i
- `dp[i] = max(dp[j] + 1)` for all j < i where `arr[j] < arr[i]`

**O(n log n) solution using Patience Sorting with binary search:**

### Java Implementation — LIS O(n log n)

```java
import java.util.*;

public class LIS {
    public int lengthOfLIS(int[] nums) {
        List<Integer> tails = new ArrayList<>();
        for (int num : nums) {
            int pos = Collections.binarySearch(tails, num);
            if (pos < 0) pos = -(pos + 1);
            if (pos == tails.size()) tails.add(num);
            else tails.set(pos, num);
        }
        return tails.size();
    }
}
```

## Coin Change Problem

Given an amount and a list of coin denominations, find the minimum number of coins needed to make the amount. Coins can be used unlimited times.

**Recurrence:** `dp[i] = min(dp[i], dp[i - coin] + 1)` for each coin ≤ i.

**Time Complexity:** O(amount × number of coins)
**Space Complexity:** O(amount)

### Java Implementation — Coin Change

```java
public class CoinChange {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
}
```

## Matrix Chain Multiplication

Given a sequence of matrices, find the most efficient way to multiply them (minimizing scalar multiplications).

**Recurrence:** `dp[i][j] = min(dp[i][k] + dp[k+1][j] + dims[i-1]*dims[k]*dims[j])` for i ≤ k < j.

**Time Complexity:** O(n³)
**Space Complexity:** O(n²)

## Memoization vs Tabulation

| Aspect | Memoization (Top-down) | Tabulation (Bottom-up) |
|--------|----------------------|----------------------|
| Approach | Recursive + cache | Iterative table fill |
| Subproblems | Only needed ones | All subproblems |
| Stack overflow | Possible for deep recursion | No risk |
| Code simplicity | Often simpler | Can be harder to see |
| Performance | Slight overhead from recursion | Generally faster |
