//
//  SceneObjects.hpp
//  RayTracer
//
//  Created by Bo Miller on 1/2/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//
#include "glm/glm/glm.hpp"
#include "glm/glm/gtx/io.hpp"
#include <stdio.h>
#include <vector>

#ifndef SceneObjects_hpp
#define SceneObjects_hpp

class Light
{
public:
    glm::vec3 color;
    glm::vec3 direction;
    bool point;
    float constantTerm;
    float linearTerm;
    float quadraticTerm;
    glm::vec3 position;
    bool area;
    float radius;
};

class SceneObject
{
public:
    int objNum;
    glm::vec3 position;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 reflective;
    float radius;
    glm::vec3 center;
    float shininess;
    bool sphere;
    bool triangle;
    glm::vec3 v1;
    glm::vec3 v1Norm;
    glm::vec3 v2;
    glm::vec3 v2Norm;
    glm::vec3 v3;
    glm::vec3 v3Norm;
};

#endif /* SceneObjects_hpp */
