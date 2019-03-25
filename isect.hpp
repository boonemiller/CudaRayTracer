//
//  isect.hpp
//  RayTracer
//
//  Created by Bo Miller on 1/2/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//
#include "glm/glm/glm.hpp"
#include "glm/glm/gtx/io.hpp"
#include <stdio.h>
#include "SceneObjects.hpp"
class Isect{
public:
    bool isected;
    glm::vec3 isectPoint;
    glm::vec3 incidentDirection;
    glm::vec3 reflectionCoef;
    //*****from intersect object
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 reflective;
    float shininess;
    //***********
    glm::vec3 normal;
    glm::vec3 color;
    int i;
    int j;
};
