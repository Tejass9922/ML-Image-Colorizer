def binarySearch(arr, x):
    low = 0
    high = len(arr) - 1
 
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid      # key found
 
    return low              # key not found
 
 
# Function to find the `k` closest elements to `x` in a sorted integer array `arr`
def findKClosestElements(arr, x, k):
 
    # find the insertion point using the binary search algorithm
    i = binarySearch(arr, x)
 
    left = i - 1
    right = i
 
    # run `k` times
    while k > 0:
 
        # compare the elements on both sides of the insertion point `i`
        # to get the first `k` closest elements
 
        if left < 0 or (right < len(arr) and abs(arr[left] - x) > abs(arr[right] - x)):
            right = right + 1
        else:
            left = left - 1
 
        k = k - 1
 
    # return `k` closest elements
    return arr[left+1: right]

arr = [1,1,1,1,2,2,2,3,4]
x = 1
k = 5
k_closest = findKClosestElements(arr,x,k)
print(k_closest)