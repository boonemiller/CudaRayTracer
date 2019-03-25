//
//  bvh.hpp
//  RayTracer
//
//  Created by Bo Miller on 2/1/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//
#include "SceneObjects.hpp"
#include <vector>
#ifndef bvh_hpp
#define bvh_hpp

class Node
{
public:
    Node* left;
    Node* right;
    bool isleaf = false;
    SceneObject objs[3];
    int numObjs;
    //some bounding box variables
    double minX;
    double maxX;
    double minY;
    double maxY;
    double minZ;
    double maxZ;
    
    double midpoint;
    double longestAxis;
};

Node* constructTree(std::vector<SceneObject>& objects, Node* currentNode);


#include <stdio.h>

#endif /* bvh_hpp */
