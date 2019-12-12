//
//  bvh.hpp
//  RayTracer
//
//  Created by Bo Miller on 2/1/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//
#include "SceneObjects.hpp"
#include <vector>
#include <deque>
#ifndef bvh_hpp
#define bvh_hpp

class Node
{
public:
    Node* parent;
    Node* left;
    Node* right;
    bool isleaf;
    SceneObject* objs[64];
    int numObjs;
    //some bounding box variables
    float minX;
    float maxX;
    float minY;
    float maxY;
    float minZ;
    float maxZ;
    int nodeNum;
    float midpoint;
    float longestAxis;
};

Node* constructTree(std::vector<SceneObject*>& objects, Node*& currentNode,std::deque<Node *>& leafs, Node*& parentNode);
void refitTree(std::deque<Node *>& leafs);
void freeTree(Node*& root);

#include <stdio.h>

#endif /* bvh_hpp */
