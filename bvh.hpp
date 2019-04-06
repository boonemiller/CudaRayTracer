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
    bool isleaf;
    SceneObject objs[6];
    int numObjs;
    //some bounding box variables
    float minX;
    float maxX;
    float minY;
    float maxY;
    float minZ;
    float maxZ;
    
    float midpoint;
    float longestAxis;
};

Node* constructTree(std::vector<SceneObject>& objects, Node* currentNode);


#include <stdio.h>

#endif /* bvh_hpp */
