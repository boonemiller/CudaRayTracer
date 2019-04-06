//
//  bvh.cpp
//  RayTracer
//
//  Created by Bo Miller on 2/1/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//
#include "bvh.hpp"
#include <limits>
Node* constructTree(std::vector<SceneObject>& objects, Node* currentNode)
{
    if(objects.size() <= 6)
    {
        for(int i = 0; i<objects.size();i++)
        {
            currentNode->objs[i] = objects[i];
        }
        currentNode->numObjs = (int)objects.size();
        currentNode->isleaf = true;
        return currentNode;
    }
    
    Node* newLeftNode;
    cudaMallocManaged(&newLeftNode,sizeof(Node));
    newLeftNode->left = NULL;
    newLeftNode->right = NULL;
    newLeftNode->isleaf = false;

    Node* newRightNode;
    cudaMallocManaged(&newRightNode,sizeof(Node));
    newRightNode->left = NULL;
    newRightNode->right = NULL;
    newRightNode->isleaf = false;
    
    std::vector<SceneObject> leftObjects;
    glm::vec3 midLeft;
    float maxLeftX = std::numeric_limits<float>::min();
    float minLeftX = std::numeric_limits<float>::max();
    float maxLeftY = std::numeric_limits<float>::min();
    float minLeftY = std::numeric_limits<float>::max();
    float maxLeftZ = std::numeric_limits<float>::min();
    float minLeftZ = std::numeric_limits<float>::max();
    glm::vec3 midRight;
    float maxRightX = std::numeric_limits<float>::min();
    float minRightX = std::numeric_limits<float>::max();
    float maxRightY = std::numeric_limits<float>::min();
    float minRightY = std::numeric_limits<float>::max();
    float maxRightZ = std::numeric_limits<float>::min();
    float minRightZ = std::numeric_limits<float>::max();
    std::vector<SceneObject> rightObjects;
    for(int i = 0; i < objects.size(); i++)
    {
        float objmidpoint;
        if(objects[i].sphere)
            objmidpoint = objects[i].position[currentNode->longestAxis];
        else
        {
            objmidpoint = (objects[i].v1[currentNode->longestAxis]+objects[i].v2[currentNode->longestAxis]+objects[i].v3[currentNode->longestAxis])/3.0f;
        }
        
        if(objmidpoint <= currentNode->midpoint)
        {
            if(objects[i].sphere)
            {
                if(objects[i].position[0]-objects[i].radius < minLeftX)
                    minLeftX = objects[i].position[0]-objects[i].radius;
                if(objects[i].position[1]-objects[i].radius < minLeftY)
                    minLeftY = objects[i].position[1]-objects[i].radius;
                if(objects[i].position[2]-objects[i].radius < minLeftZ)
                    minLeftZ = objects[i].position[2]-objects[i].radius;
                
                if(objects[i].position[0]+objects[i].radius > maxLeftX)
                    maxLeftX = objects[i].position[0]+objects[i].radius;
                if(objects[i].position[1]+objects[i].radius > maxLeftY)
                    maxLeftY = objects[i].position[1]+objects[i].radius;
                if(objects[i].position[2]+objects[i].radius > maxLeftZ)
                    maxLeftZ = objects[i].position[2]+objects[i].radius;
                midLeft += objects[i].position;
            }
            else if(objects[i].triangle)
            {
                if(objects[i].v1[0] < minLeftX)
                    minLeftX = objects[i].v1[0];
                if(objects[i].v1[1] < minLeftY)
                    minLeftY = objects[i].v1[1];
                if(objects[i].v1[2] < minLeftZ)
                    minLeftZ = objects[i].v1[2];
                
                if(objects[i].v1[0] > maxLeftX)
                    maxLeftX = objects[i].v1[0];
                if(objects[i].v1[1] > maxLeftY)
                    maxLeftY = objects[i].v1[1];
                if(objects[i].v1[2] > maxLeftZ)
                    maxLeftZ = objects[i].v1[2];
                
                if(objects[i].v2[0] < minLeftX)
                    minLeftX = objects[i].v2[0];
                if(objects[i].v2[1] < minLeftY)
                    minLeftY = objects[i].v2[1];
                if(objects[i].v2[2] < minLeftZ)
                    minLeftZ = objects[i].v2[2];
                
                if(objects[i].v2[0] > maxLeftX)
                    maxLeftX = objects[i].v2[0];
                if(objects[i].v2[1] > maxLeftY)
                    maxLeftY = objects[i].v2[1];
                if(objects[i].v2[2] > maxLeftZ)
                    maxLeftZ = objects[i].v2[2];
                
                if(objects[i].v3[0] < minLeftX)
                    minLeftX = objects[i].v3[0];
                if(objects[i].v3[1] < minLeftY)
                    minLeftY = objects[i].v3[1];
                if(objects[i].v3[2] < minLeftZ)
                    minLeftZ = objects[i].v3[2];
                
                if(objects[i].v3[0] > maxLeftX)
                    maxLeftX = objects[i].v3[0];
                if(objects[i].v3[1] > maxLeftY)
                    maxLeftY = objects[i].v3[1];
                if(objects[i].v3[2] > maxLeftZ)
                    maxLeftZ = objects[i].v3[2];
                midLeft += glm::vec3((objects[i].v1[0]+objects[i].v2[0]+objects[i].v3[0])/3.0f,(objects[i].v1[1]+objects[i].v2[1]+objects[i].v3[1])/3.0f,(objects[i].v1[2]+objects[i].v2[2]+objects[i].v3[2])/3.0f);
            }
            leftObjects.push_back(objects[i]);
        }
        else
        {
            if(objects[i].sphere)
            {
                if(objects[i].position[0]-objects[i].radius < minRightX)
                    minRightX = objects[i].position[0]-objects[i].radius;
                if(objects[i].position[1]-objects[i].radius < minRightY)
                    minRightY = objects[i].position[1]-objects[i].radius;
                if(objects[i].position[2]-objects[i].radius < minRightZ)
                    minRightZ = objects[i].position[2]-objects[i].radius;
                
                if(objects[i].position[0]+objects[i].radius > maxRightX)
                    maxRightX = objects[i].position[0]+objects[i].radius;
                if(objects[i].position[1]+objects[i].radius > maxRightY)
                    maxRightY = objects[i].position[1]+objects[i].radius;
                if(objects[i].position[2]+objects[i].radius > maxRightZ)
                    maxRightZ = objects[i].position[2]+objects[i].radius;
                midRight += objects[i].position;
            }
            else if(objects[i].triangle)
            {
                if(objects[i].v1[0] < minRightX)
                    minRightX = objects[i].v1[0];
                if(objects[i].v1[1] < minRightY)
                    minRightY = objects[i].v1[1];
                if(objects[i].v1[2] < minRightZ)
                    minRightZ = objects[i].v1[2];
                
                if(objects[i].v1[0] > maxRightX)
                    maxRightX = objects[i].v1[0];
                if(objects[i].v1[1] > maxRightY)
                    maxRightY = objects[i].v1[1];
                if(objects[i].v1[2] > maxRightZ)
                    maxRightZ = objects[i].v1[2];
                
                if(objects[i].v2[0] < minRightX)
                    minRightX = objects[i].v2[0];
                if(objects[i].v2[1] < minRightY)
                    minRightY = objects[i].v2[1];
                if(objects[i].v2[2] < minRightZ)
                    minRightZ = objects[i].v2[2];
                
                if(objects[i].v2[0] > maxRightX)
                    maxRightX = objects[i].v2[0];
                if(objects[i].v2[1] > maxRightY)
                    maxRightY = objects[i].v2[1];
                if(objects[i].v2[2] > maxRightZ)
                    maxRightZ = objects[i].v2[2];
                
                if(objects[i].v3[0] < minRightX)
                    minRightX = objects[i].v3[0];
                if(objects[i].v3[1] < minRightY)
                    minRightY = objects[i].v3[1];
                if(objects[i].v3[2] < minRightZ)
                    minRightZ = objects[i].v3[2];
                
                if(objects[i].v3[0] > maxRightX)
                    maxRightX = objects[i].v3[0];
                if(objects[i].v3[1] > maxRightY)
                    maxRightY = objects[i].v3[1];
                if(objects[i].v3[2] > maxRightZ)
                    maxRightZ = objects[i].v3[2];
                midRight += glm::vec3((objects[i].v1[0]+objects[i].v2[0]+objects[i].v3[0])/3.0f,(objects[i].v1[1]+objects[i].v2[1]+objects[i].v3[1])/3.0f,(objects[i].v1[2]+objects[i].v2[2]+objects[i].v3[2])/3.0f);
            }
            rightObjects.push_back(objects[i]);
        }
    }
    
    midLeft = glm::vec3(midLeft[0]/leftObjects.size(),midLeft[1]/leftObjects.size(),midLeft[2]/leftObjects.size());
    midRight = glm::vec3(midRight[0]/rightObjects.size(),midRight[1]/rightObjects.size(),midRight[2]/rightObjects.size());
    
    if(maxLeftX-minLeftX > maxLeftY - minLeftY)
    {
        if(maxLeftX-minLeftX > maxLeftZ - minLeftZ)
        {
            newLeftNode->longestAxis = 0;
            newLeftNode->midpoint = midLeft[0];
        }
    }
    if(maxLeftY-minLeftY > maxLeftX - minLeftX)
    {
        if(maxLeftY-minLeftY > maxLeftZ - minLeftZ)
        {
            newLeftNode->longestAxis = 1;
            newLeftNode->midpoint = midLeft[1];
        }
    }
    if(maxLeftZ - minLeftZ > maxLeftX - minLeftX)
    {
        if(maxLeftZ - minLeftZ > maxLeftY-minLeftY)
        {
            newLeftNode->longestAxis = 2;
            newLeftNode->midpoint = midLeft[2];
        }
    }
    
    if(maxRightX-minRightX > maxRightY - minRightY)
    {
        if(maxRightX-minRightX > maxRightZ - minRightZ)
        {
            newRightNode->longestAxis = 0;
            newRightNode->midpoint = midRight[0];
        }
    }
    if(maxRightY-minRightY > maxRightX - minRightX)
    {
        if(maxRightY-minRightY > maxRightZ - minRightZ)
        {
            newRightNode->longestAxis = 1;
            newRightNode->midpoint = midRight[1];
        }
    }
    if(maxRightZ-minRightZ > maxRightX - minRightX)
    {
        if(maxRightZ-minRightZ > maxRightY - minRightY)
        {
            newRightNode->longestAxis = 2;
            newRightNode->midpoint = midRight[2];
        }
    }
    newLeftNode->minX = minLeftX;
    newLeftNode->maxX = maxLeftX;
    newLeftNode->minY = minLeftY;
    newLeftNode->maxY = maxLeftY;
    newLeftNode->minZ = minLeftZ;
    newLeftNode->maxZ = maxLeftZ;
    
    newRightNode->minX = minRightX;
    newRightNode->maxX = maxRightX;
    newRightNode->minY = minRightY;
    newRightNode->maxY = maxRightY;
    newRightNode->minZ = minRightZ;
    newRightNode->maxZ = maxRightZ;
    
    currentNode->left  = constructTree(leftObjects, newLeftNode);
    currentNode->right = constructTree(rightObjects, newRightNode);
    
    return currentNode;
}



