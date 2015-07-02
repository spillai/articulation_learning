#ifndef OPENCV_MATH_UTILS_H__
#define OPENCV_MATH_UTILS_H__

#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace opencv_utils { 
double l2_dist(const cv::Point& p1, const cv::Point& p2);
double l2_dist(const cv::Point2f& p1, const cv::Point2f& p2);
double l2_dist(const cv::Point3f& p1, const cv::Point3f& p2);
// template<typename _Tp> double l2_dist(const cv::Point_<_Tp>& p1, const cv::Point_<_Tp>& p2);
// template<typename _Tp> inline double l2_dist(const cv::Point3_<_Tp>& p1, const cv::Point3_<_Tp>& p2);

cv::Mat compute_1D_histogram(cv::Mat& img, int nbins=10, const cv::Mat& mask = cv::Mat());

// Compute the mean of the contour
cv::Point2f mean_contour(const std::vector<cv::Point>& pts);
cv::Point2f mean_contour(const std::vector<cv::KeyPoint>& kpts);

// Shrink contour for stable depth features
std::vector<cv::Point> scale_contour(const std::vector<cv::Point>& pts, float scale=1.2);

// This function splits the input sequence or set into one or more equivalence classes and
// returns the vector of labels - 0-based class indexes for each element.
// predicate(a,b) returns true if the two sequence elements certainly belong to the same class.
//
// The algorithm is described in "Introduction to Algorithms"
// by Cormen, Leiserson and Rivest, the chapter "Data structures for disjoint sets"
template<typename _Tp, class _EqPredicate> int
partition( const std::vector<_Tp>& _vec, std::vector<int>& labels,
           _EqPredicate predicate=_EqPredicate())
{
  int i, j, N = (int)_vec.size();
  const _Tp* vec = &_vec[0];

  const int PARENT=0;
  const int RANK=1;

  std::vector<int> _nodes(N*2);
  int (*nodes)[2] = (int(*)[2])&_nodes[0];

  // The first O(N) pass: create N single-vertex trees
  for(i = 0; i < N; i++)
  {
    nodes[i][PARENT]=-1;
    nodes[i][RANK] = 0;
  }

  // The main O(N^2) pass: merge connected components
  for( i = 0; i < N; i++ )
  {
    int root = i;

    // find root
    while( nodes[root][PARENT] >= 0 )
      root = nodes[root][PARENT];

    for( j = 0; j < N; j++ )
    {
      if( i == j || !predicate(vec[i], vec[j]))
        continue;
      int root2 = j;

      while( nodes[root2][PARENT] >= 0 )
        root2 = nodes[root2][PARENT];

      if( root2 != root )
      {
        // unite both trees
        int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
        if( rank > rank2 )
          nodes[root2][PARENT] = root;
        else
        {
          nodes[root][PARENT] = root2;
          nodes[root2][RANK] += rank == rank2;
          root = root2;
        }
        CV_Assert( nodes[root][PARENT] < 0 );

        int k = j, parent;

        // compress the path from node2 to root
        while( (parent = nodes[k][PARENT]) >= 0 )
        {
          nodes[k][PARENT] = root;
          k = parent;
        }

        // compress the path from node to root
        k = i;
        while( (parent = nodes[k][PARENT]) >= 0 )
        {
          nodes[k][PARENT] = root;
          k = parent;
        }
      }
    }
  }

  // Final O(N) pass: enumerate classes
  labels.resize(N);
  int nclasses = 0;

  for( i = 0; i < N; i++ )
  {
    int root = i;
    while( nodes[root][PARENT] >= 0 )
      root = nodes[root][PARENT];
    // re-use the rank as the class label
    if( nodes[root][RANK] >= 0 )
      nodes[root][RANK] = ~nclasses++;
    labels[i] = ~nodes[root][RANK];
  }

  return nclasses;
}


}

#endif // OPENCV_MATH_UTILS_H
