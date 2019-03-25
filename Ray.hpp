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
    bool validRay; 
    glm::vec3 position;
    glm::vec3 direction;
    std::pair<int,int> pixel;
    int i;
    int j;
    glm::vec3 color;
    int timesbounced;
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

void startRayTracing(float width, float height, float (&pixelcolorBuffer)[360][720][3],glm::vec3 cameraPosition, glm::vec3 cameraDirection, std::vector<SceneObject>& scene, std::vector<Light>& lights, Node* rootnode);
void* CastRays(void *arguments);
//bool boundingBoxIntersection(glm::vec3 position, glm::vec3 direction, Node* box);
bool intersectSphere(glm::vec3 position, glm::vec3 direction, SceneObject s, float& iTime,glm::vec3& normal, glm::vec3& intersection);
glm::vec3 checkLights(Ray& r, glm::vec3 normal, glm::vec3 intersection, std::vector<Light>& lights, float t, SceneObject s, std::vector<SceneObject>& scene);
bool rayPlaneIntersection(Ray &r, std::vector<SceneObject>& scene, std::vector<Light>& lights, int numBounces, Ray& reflectedRay);
bool intersectTriangle(glm::vec3 v1,glm::vec3 v2,glm::vec3 v3,glm::vec3 position, glm::vec3 direction, glm::vec3& n, glm::vec3& intersection, float& time);
bool intersectObjects(Ray &r, std::vector<SceneObject>& scene, std::vector<Light>& lights, int numBounces, float& t, Ray& reflectedRay);
#endif /* Ray_hpp */
