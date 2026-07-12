# Sorting Algorithms

## Merge Sort

Merge Sort is a divide-and-conquer sorting algorithm that recursively splits an array in half, sorts each half, then merges the sorted halves.

**Time Complexity:**
- Best case: O(n log n)
- Average case: O(n log n)
- Worst case: O(n log n)

**Space Complexity:** O(n) auxiliary space for the merge step.

Merge Sort is stable (preserves relative order of equal elements) and is preferred when stability is required or when sorting linked lists.

### Java Implementation

```java
public class MergeSort {
    public void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    private void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        int[] L = new int[n1];
        int[] R = new int[n2];
        System.arraycopy(arr, left, L, 0, n1);
        System.arraycopy(arr, mid + 1, R, 0, n2);
        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) arr[k++] = L[i++];
            else arr[k++] = R[j++];
        }
        while (i < n1) arr[k++] = L[i++];
        while (j < n2) arr[k++] = R[j++];
    }
}
```

## Quick Sort

Quick Sort selects a pivot element and partitions the array so all elements less than the pivot are to its left and all greater are to its right, then recursively sorts both partitions.

**Time Complexity:**
- Best case: O(n log n) — pivot consistently near median
- Average case: O(n log n)
- Worst case: O(n²) — pivot is always smallest or largest (sorted/reverse-sorted input)

**Space Complexity:** O(log n) average (stack depth), O(n) worst case.

Quick Sort is NOT stable. It is typically faster in practice than Merge Sort due to better cache locality.

### Java Implementation (Lomuto Partition)

```java
public class QuickSort {
    public void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    private int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
            }
        }
        int temp = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = temp;
        return i + 1;
    }
}
```

## Heap Sort

Heap Sort builds a max-heap from the array, then repeatedly extracts the maximum to produce a sorted array.

**Time Complexity:**
- All cases: O(n log n)

**Space Complexity:** O(1) — sorts in place.

Heap Sort is NOT stable. It is often used when guaranteed O(n log n) with O(1) space is required.

### Java Implementation

```java
public class HeapSort {
    public void heapSort(int[] arr) {
        int n = arr.length;
        for (int i = n / 2 - 1; i >= 0; i--) heapify(arr, n, i);
        for (int i = n - 1; i > 0; i--) {
            int temp = arr[0]; arr[0] = arr[i]; arr[i] = temp;
            heapify(arr, i, 0);
        }
    }

    private void heapify(int[] arr, int n, int i) {
        int largest = i, left = 2 * i + 1, right = 2 * i + 2;
        if (left < n && arr[left] > arr[largest]) largest = left;
        if (right < n && arr[right] > arr[largest]) largest = right;
        if (largest != i) {
            int swap = arr[i]; arr[i] = arr[largest]; arr[largest] = swap;
            heapify(arr, n, largest);
        }
    }
}
```

## Comparison of Sorting Algorithms

| Algorithm   | Best     | Average  | Worst    | Space   | Stable |
|-------------|----------|----------|----------|---------|--------|
| Merge Sort  | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort  | O(n log n) | O(n log n) | O(n²)  | O(log n) | No  |
| Heap Sort   | O(n log n) | O(n log n) | O(n log n) | O(1) | No  |
| Insertion Sort | O(n) | O(n²)  | O(n²)  | O(1)    | Yes    |
| Bubble Sort | O(n)  | O(n²)  | O(n²)  | O(1)    | Yes    |

For small arrays (n < 20), Insertion Sort often outperforms asymptotically better algorithms due to low constant factors.
