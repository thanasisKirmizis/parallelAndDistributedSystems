#ifndef VPTREE_H
#define VPTREE_H

/**
* A Tree Node Structure
*
* @param vp: The Vantage point
* @param md: The median distance of the vantage point to the other points 
* @param idx: The index of the vantage point in the original set
* @param innerIdxs: A pointer to the indexes (in the initial dataset) of the points of the inner sub-tree
* @param outerIdxs: A pointer to the indexes (in the initial dataset) of the points of the outer sub-tree
* @param numOfPointsInSubtree: Number of points in the subtree below this node (including this node)
* 
**/

struct vptree{
	double *vp;
	double md;
	int idx;
	int innerSplitPtr;
	int outerSplitPtr;
	int numOfPointsInSubtree;
};

typedef struct vptree vptree;

#endif
