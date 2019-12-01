#ifndef AUX_H
#define AUX_H

#define SWAP1(x, y) { double temp = x; x = y; y = temp; }
#define SWAP2(x, y) { int temp = x; x = y; y = temp; }

//The partioning function to be used in double concurrent quickselect
int partition(double *a, int *b, int left, int right, int pivotIndex)
{
	//Pick pivotIndex as pivot from the array
	double pivot = a[pivotIndex];

	//Move pivot to end
	SWAP1(a[pivotIndex], a[right]);
	SWAP2(b[pivotIndex], b[right]);

	//Elements less than pivot will be pushed to the left of pIndex
	//Elements more than pivot will be pushed to the right of pIndex
	//Equal elements can go either way
	int pIndex = left;
	int i;

	//Each time we finds an element less than or equal to pivot, pIndex
	//is incremented and that element would be placed before the pivot.
	for (i = left; i < right; i++)
	{
		if (a[i] <= pivot)
		{
			SWAP1(a[i], a[pIndex]);
			SWAP2(b[i], b[pIndex]);
			pIndex++;
		}
	}

	//Move pivot to its final place
	SWAP1(a[pIndex], a[right]);
	SWAP2(b[pIndex], b[right]);
	
	//Return pIndex (index of pivot element)
	return pIndex;
}

//A function to implement double concurrent quickselect
double quickselect(double *A, int *B, int left, int right, int k)
{
	//If the array contains only one element, return that element
	if (left == right)
		return A[left];

	//Select a pivotIndex between left and right
	int pivotIndex = left + rand() % (right - left + 1);

	pivotIndex = partition(A, B, left, right, pivotIndex);

	//The pivot is in its final sorted position
	if (k == pivotIndex)
		return A[k];

	//If k is less than the pivot index
	else if (k < pivotIndex)
		return quickselect(A, B, left, pivotIndex - 1, k);

	//If k is more than the pivot index
	else
		return quickselect(A, B, pivotIndex + 1, right, k);
}

//A function to implement double concurrent bubble sort 
void doubleBubbleSort(double *a, int *b, int n) { 
	   
	for (int i = 0; i < n-1; i++) {    
	   for (int j = 0; j < n-i-1; j++) {
		   
		   if (a[j] > a[j+1]) {
			   
				SWAP1(a[j],a[j+1])
				SWAP2(b[j],b[j+1])
		   }
	   }
	}
}

#endif
