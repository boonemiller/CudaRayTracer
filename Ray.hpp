//
//  Ray.hpp
//  RayTracer
//
//  Created by Bo Miller on 1/2/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//

#include "SceneObjects.hpp"
#include "bvh.hpp"
#ifndef Ray_hpp
#define Ray_hpp

#include <stdio.h>

class Ray{
public:
    glm::vec3 position;
    glm::vec3 direction;
    std::pair<int,int> pixel;
    int i;
    int j;
    glm::vec3 color;
    //if its a reflection ray we need to know the reflective constant of the surface we are reflecting off of
    glm::vec3 surfaceReflectiveCoef;
    /*
     Ray Type:
     Primary Ray: 0
     Secondary/reflective Ray: 1
     Shadow Ray: 2
     */
    int raytype;
};

void startRayTracing(float width, float height,  unsigned char*& pixelcolorBuffer,glm::vec3 cameraPosition, glm::vec3 cameraDirection, std::vector<SceneObject *>& scene, std::vector<Light>& lights, Node* rootnode);
#endif /* Ray_hpp */
