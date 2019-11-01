#ifndef VPTREE_H
#define VPTREE_H

/**
* A Tree Node Structure
*
* @param vp: The Vantage point
* @param md: The median distance of the vantage point to the other points 
* @param idx: The index of the vantage point in the original set
* @param inner: A pointer to the inner sub-tree
* @param outer: A pointer to the outer sub-tree
* 
**/
struct vptree{
	double *vp;
	double md;
	int idx;
	struct vptree *inner;
	struct vptree *outer;
};

typedef struct vptree vptree;

// ========== LIST OF ACCESSORS

//! Build vantage-point tree given input dataset X
/*!
\param X Input data points, stored as [n-by-d] array
\param n Number of data points (rows of X)
\param d Number of dimensions (columns of X)
\return The vantage-point tree
*/

struct vptree * buildvp(double *X, int n, int d);

struct vptree * vpt(int n, int dim, double *x, int idxs[]);

//! Return vantage-point subtree with points inside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/

struct vptree * getInner(struct vptree * T);


//! Return vantage-point subtree with points outside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/

struct vptree * getOuter(struct vptree * T);


//! Return median of distances to vantage point
/*!
\param node A vantage-point tree
\return The median distance
*/

double getMD(struct vptree * T);


//! Return the coordinates of the vantage point
/*!
\param node A vantage-point tree
\return The coordinates [d-dimensional vector]
*/

double * getVP(struct vptree * T);


//! Return the index of the vantage point
/*!
\param node A vantage-point tree
\return The index to the input vector of data points
*/

int getIDX(struct vptree * T);

#endif
