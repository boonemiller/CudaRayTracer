//
//  Ray.cpp
//  RayTracer
//
//  Created by Bo Miller on 1/2/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//LDFLAGS= -L./glm/glm -glm
#include "glm/glm/glm.hpp"
#include "glm/glm/gtx/io.hpp"
#include <iostream>
#include "Ray.hpp"
#include <vector>
#include <atomic>
#include <mutex>
#include <math.h>
#define PI 3.14159265359
#include <pthread.h>
#include <chrono>
#include "bvh.hpp"
#include <random>
#include <queue>
#include "isect.hpp"


float RAY_EPSILON = 0.000000001;
int antialiasing = 0;
int numBounces = 2;
int SampPerPix = 4;


__global__ void GeneratePrimaryRays(Ray* rays, int n, glm::vec3 L, glm::vec3 u, glm::vec3 v, glm::vec3 cameraPosition)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i+=stride)
    {
	    int ipix = i%720;
	    int jpix = i/720;
        glm::vec3 pix = (L+u*float(ipix)+v*float(jpix));

        float xoffset = 0;
        float yoffset = 0;
        glm::vec3 sample = glm::normalize(glm::vec3(pix[0]+xoffset,pix[1]+yoffset,pix[2])-cameraPosition);
        rays[i].raytype = 0;
        rays[i].position = cameraPosition;
        rays[i].direction = sample;
        rays[i].i = ipix;
        rays[i].j = jpix;
        rays[i].color = glm::vec3(0,0,0);
        rays[i].surfaceReflectiveCoef = glm::vec3(0,0,0);
    }
}

__device__ bool boundingBoxIntersection(glm::vec3& position, glm::vec3& direction, Node* node)
{
    float tmin = (node->minX-position[0])/direction[0];
    float tmax = (node->maxX-position[0])/direction[0];
    
    if(tmin>tmax)
    {
        float temp = tmin;
        tmin = tmax;
        tmax = temp;
    }
    float tymin = (node->minY-position[1])/direction[1];
    float tymax = (node->maxY-position[1])/direction[1];
    
    if(tymin>tymax)
    {
        float temp = tymin;
        tymin = tymax;
        tymax = temp;
    }
    
    if((tmin > tymax) || (tymin > tmax))
        return false;
    
    if (tymin > tmin)
        tmin = tymin;
    
    if (tymax < tmax)
        tmax = tymax;
    
    float tzmin = (node->minZ-position[2])/direction[2];
    float tzmax = (node->maxZ-position[2])/direction[2];
    
    if (tzmin > tzmax)
    {
        float temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }
    
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    
    if (tzmin > tmin)
        tmin = tzmin;
    
    if (tzmax < tmax)
        tmax = tzmax;
    
    return true;
}
__device__ bool intersectSphere(SceneObject& s, glm::vec3& position, glm::vec3& direction, float& isectT, glm::vec3& normal, glm::vec3& intersection)
{
    float RAY_EPSILON = 0.000000001;
    float a = glm::dot(direction, direction);
    float b = 2 * glm::dot(direction,position-s.position);
    float c = glm::dot(s.position,s.position) + glm::dot(position,position) + (-2 * glm::dot(s.position,position)) - pow(s.radius,2);
    
    float discriminant = b*b - 4*a*c;
    
    if(discriminant > 0.0+RAY_EPSILON)
    {
    
        float t = (-b - sqrt(discriminant))/(2*a);
    
        float t2 = (-b + sqrt(discriminant))/(2*a);
    
    
        if(t2>RAY_EPSILON)
        {
            //we know we have some intersection
    
            if( t > RAY_EPSILON )
            {
                isectT = t;
            }
            else
            {
                isectT = t2;
            }
                
            intersection = position+t*direction;
            normal = glm::normalize((intersection-s.position)/s.radius);
            return true;
        }
    }
    return false;
}

__device__ bool intersectTriangle(SceneObject& s, glm::vec3& position, glm::vec3& direction, glm::vec3& n, glm::vec3& intersection, float& time)
{
    glm::vec3 normal = glm::normalize(glm::cross((s.v2-s.v1),(s.v3-s.v1)));
    
    float denom = glm::dot(normal,direction);
    if(abs(denom) > .0001f)
    {
        float t = glm::dot((s.v1-position),normal)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = position+t*direction;
            
            float test1 = glm::dot(glm::cross((s.v2-s.v1),(intersect-s.v1)),normal);
            float test2 = glm::dot(glm::cross((s.v3-s.v2),(intersect-s.v2)),normal);
            float test3 = glm::dot(glm::cross((s.v1-s.v3),(intersect-s.v3)),normal);
            if(test1 >= 0.0 && test2 >= 0.0 && test3 >= 0.0)
            {
                glm::vec3 v0 = s.v2 - s.v1; 
                glm::vec3 v1 = s.v3 - s.v1;
                glm::vec3 v2 = intersect - s.v1;
                float d00 = glm::dot(v0, v0);
                float d01 = glm::dot(v0, v1);
                float d11 = glm::dot(v1, v1);
                float d20 = glm::dot(v2, v0);
                float d21 = glm::dot(v2, v1);
                float denom = d00 * d11 - d01 * d01;
                float v = (d11 * d20 - d01 * d21) / denom;
                float w = (d00 * d21 - d01 * d20) / denom;
                float u = 1.0f - v - w;
                
                n = glm::normalize(u*s.v1Norm+v*s.v2Norm+w*s.v3Norm);
		//n = -normal;
                intersection = intersect;
                time = t;
                return true;
            }
        }
    }
    return false;
}

__device__ void bvhTraverse(glm::vec3& position, glm::vec3& direction, Node* currentNode,bool& intersect,float& minT, SceneObject& intersectObj, glm::vec3& minTnormal, glm::vec3& minTintersection)
{
    if(currentNode->isleaf)
    {
        if(boundingBoxIntersection(position, direction, currentNode))
        {
            for(int i = 0; i<currentNode->numObjs;i++)
            {
                if(currentNode->objs[i]->sphere)
                {
                    float iTime;
                    glm::vec3 normal;
                    glm::vec3 intersection;
                    if(intersectSphere(*currentNode->objs[i], position, direction, iTime, normal, intersection))
                    {
                        if(iTime<minT)
                        {
                            minTnormal = normal;
                            minTintersection = intersection;
                            intersectObj = *currentNode->objs[i];
                            minT = iTime;
                            intersect = true;
                        }
                    }
                }
                else if(currentNode->objs[i]->triangle)
                {
                    float intersectT;
                    glm::vec3 normal;
                    glm::vec3 intersection;
                    if(intersectTriangle(*currentNode->objs[i], position, direction, normal, intersection, intersectT))
                    {
                        if(intersectT<minT)
                        {
                            minTnormal = normal;
                            minTintersection = intersection;
                            intersectObj = *currentNode->objs[i];
                            minT = intersectT;
                            intersect = true;
                        }
                    }
                }   
            }
        }
    }
    else
    {
        if(boundingBoxIntersection(position, direction, currentNode->left))
            bvhTraverse(position, direction,currentNode->left,intersect,minT,intersectObj,minTnormal,minTintersection);
        if(boundingBoxIntersection(position, direction, currentNode->right))
            bvhTraverse(position, direction,currentNode->right,intersect,minT,intersectObj,minTnormal,minTintersection);
    }
}

__device__ bool wallIntersection(Isect& ipoint, Ray& r, Ray& reflect, Node* root)
{
    float x = 200.0f;
    float y = 200.0f;
    float z = 300.0f;
    glm::vec3 up = glm::vec3(0,1,0);
    float denom = glm::dot(up,r.direction);
    if(fabsf(denom) > .0001f)
    {
        float t = glm::dot((glm::vec3(0,-y,0)-r.position),up)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = r.position+t*r.direction;
            
            if(intersect[2]>-z && intersect[2]<z && intersect[0]>-x && intersect[0] < x)
            {
                if(r.raytype == 0)
                {
                    ipoint.color = 0.2f * glm::vec3(1.0, 0.2, 0.2);;
                    reflect.surfaceReflectiveCoef =  glm::vec3(0.0,0.0,0.0);
                    ipoint.reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                }
                else if(r.raytype == 1){
                    ipoint.reflectionCoef = r.surfaceReflectiveCoef;
                    reflect.surfaceReflectiveCoef = r.surfaceReflectiveCoef* glm::vec3(0.0,0.0,0.0);
                }
                ipoint.normal = up;
                ipoint.isectPoint = intersect;
                ipoint.incidentDirection = glm::normalize(glm::reflect(r.direction, up));
                ipoint.diffuse = glm::vec3(1.0, 0.2, 0.2);
                ipoint.ambient =  glm::vec3(1.0, 0.2, 0.2);
                ipoint.specular = glm::vec3(0.0,0.0,0.0);
                ipoint.shininess = 2;
                ipoint.reflective =  glm::vec3(0.0,0.0,0.0);
                return true;
            }
        }
    }
    
    
    //left wall
    up = glm::vec3(1,0,0);
    denom = glm::dot(up,r.direction);
    if(abs(denom) > .0001f)
    {
        float t = glm::dot((glm::vec3(-x,0,0)-r.position),up)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = r.position+t*r.direction;
            
            
            if(intersect[2]>-z && intersect[2]< z && intersect[1] < y && intersect[1] > -y)
            {
                if(r.raytype == 0)
                {
                    ipoint.color = 0.2f * glm::vec3(0.2, 0.2, 1.0);
                    reflect.surfaceReflectiveCoef = glm::vec3(0.0,0.0,0.0);
                    ipoint.reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                }
                else if(r.raytype == 1){
                    ipoint.reflectionCoef = r.surfaceReflectiveCoef;
                    reflect.surfaceReflectiveCoef = r.surfaceReflectiveCoef*glm::vec3(0.0,0.0,0.0);
                }
                ipoint.normal = up;
                ipoint.isectPoint = intersect;
                ipoint.incidentDirection = glm::normalize(glm::reflect(r.direction, up));
                ipoint.diffuse = glm::vec3(0.2, 0.2, 1.0);
                ipoint.ambient = glm::vec3(0.2, 0.2, 1.0);
                ipoint.specular = glm::vec3(0.0,0.0,0.0);
                ipoint.shininess = 2;
                ipoint.reflective = glm::vec3(0.0,0.0,0.0);
                return true;
            }
        }
    }
    
    //front wall, green wall in front of camera
    up = glm::vec3(0,0,1);
    denom = glm::dot(up,r.direction);
    if(abs(denom) > .0001f)
    {
        float t = glm::dot((glm::vec3(0,0,-z)-r.position),up)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = r.position+t*r.direction;
            
            if(intersect[0] > -x && intersect[0] < x && intersect[1] < y && intersect[1] > -y )
            {
                if(r.raytype == 0)
                {
                    ipoint.color = 0.2f * glm::vec3(0.0, 1.0, 0.0);
                    reflect.surfaceReflectiveCoef = glm::vec3(0.0,0.0,0.0);
                    ipoint.reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                }
                else if(r.raytype == 1){
                    ipoint.reflectionCoef = r.surfaceReflectiveCoef;
                    reflect.surfaceReflectiveCoef = r.surfaceReflectiveCoef*glm::vec3(0.0,0.0,0.0);
                }
                ipoint.normal = up;
                ipoint.isectPoint = intersect;
                ipoint.incidentDirection = glm::normalize(glm::reflect(r.direction, up));
                ipoint.diffuse = glm::vec3(0.0, 1.0, 0.0);
                ipoint.ambient = glm::vec3(0.0, 1.0, 0.0);
                ipoint.specular = glm::vec3(0.0,0.0,0.0);
                ipoint.shininess = 2;
                ipoint.reflective = glm::vec3(0.0,0.0,0.0);
                return true;
            }
        }
    }
    //back wall, yellow wall behind camera
    up = glm::vec3(0,0,-1);
    denom = glm::dot(up,r.direction);
    if(abs(denom) > .0001f)
    {
        float t = glm::dot((glm::vec3(0,0,z)-r.position),up)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = r.position+t*r.direction;
            
            if(intersect[0] > -x && intersect[0] < x && intersect[1] < y && intersect[1] > -y )
            {
                if(r.raytype == 0)
                {
                    ipoint.color = 0.2f * glm::vec3(1.0, 1.0, 0.0);
                    reflect.surfaceReflectiveCoef = glm::vec3(0.0,0.0,0.0);
                    ipoint.reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                }
                else if(r.raytype == 1){
                    ipoint.reflectionCoef = r.surfaceReflectiveCoef;
                    reflect.surfaceReflectiveCoef = r.surfaceReflectiveCoef*glm::vec3(0.0,0.0,0.0);
                }
                ipoint.normal = up;
                ipoint.isectPoint = intersect;
                ipoint.incidentDirection = glm::normalize(glm::reflect(r.direction, up));
                ipoint.diffuse =  glm::vec3(1.0, 1.0, 0.0);
                ipoint.ambient = glm::vec3(1.0, 1.0, 0.0);
                ipoint.specular = glm::vec3(0.0,0.0,0.0);
                ipoint.shininess = 2;
                ipoint.reflective = glm::vec3(0.0,0.0,0.0);
                
                return true;
            }
        }
    }
    
    //right wall
    up = glm::vec3(-1,0,0);
    denom = glm::dot(up,r.direction);
    if(abs(denom) > .0001f)
    {
        float t = glm::dot((glm::vec3(x,0,0)-r.position),up)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = r.position+t*r.direction;
            
            if(intersect[2]>-z && intersect[2]<z && intersect[1] < y && intersect[1] > -y)
            {
                if(r.raytype == 0)
                {
                    ipoint.color = 0.2f * glm::vec3(0.2, 0.2, 1.0);
                    reflect.surfaceReflectiveCoef = glm::vec3(0.0,0.0,0.0);
                    ipoint.reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                }
                else if(r.raytype == 1){
                    ipoint.reflectionCoef = r.surfaceReflectiveCoef;
                    reflect.surfaceReflectiveCoef = r.surfaceReflectiveCoef*glm::vec3(0.0,0.0,0.0);
                }
                ipoint.normal = up;
                ipoint.isectPoint = intersect;
                ipoint.incidentDirection = glm::normalize(glm::reflect(r.direction, up));
                ipoint.diffuse = glm::vec3(0.2, 0.2, 1.0);
                ipoint.ambient = glm::vec3(0.2, 0.2, 1.0);
                ipoint.specular = glm::vec3(0.0,0.0,0.0);
                ipoint.shininess = 2;
                ipoint.reflective = glm::vec3(0.0,0.0,0.0);
                return true;
            }
        }
    }
    
    //ceiling
    up = glm::vec3(0,-1,0);
    denom = glm::dot(up,r.direction);
    if(abs(denom) > .0001f)
    {
        float t = glm::dot((glm::vec3(0,y,0)-r.position), up)/denom;
        if(t >= 0.0-.0001f)
        {
            glm::vec3 intersect = r.position+t*r.direction;
            
            
            if(intersect[2] > -z && intersect[2] < z && intersect[0] > -x && intersect[0] < x)
            {
                if(r.raytype == 0)
                {
                    ipoint.color = 0.2f * glm::vec3(1, 1, 1);
                    reflect.surfaceReflectiveCoef = glm::vec3(0.0,0.0,0.0);
                    ipoint.reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                }
                else if(r.raytype == 1){
                    ipoint.reflectionCoef = r.surfaceReflectiveCoef;
                    reflect.surfaceReflectiveCoef = r.surfaceReflectiveCoef*glm::vec3(0.0,0.0,0.0);
                }
                
                ipoint.normal = up;
                ipoint.isectPoint = intersect;
                ipoint.incidentDirection = glm::normalize(glm::reflect(r.direction, up));
                ipoint.diffuse = glm::vec3(.9, .9, .9);
                ipoint.ambient = glm::vec3(1, 1,1);
                ipoint.specular = glm::vec3(0.0,0.0,0.0);
                ipoint.shininess = 2;
                ipoint.reflective = glm::vec3(0.0,0.0,0.0);
                return true;
            }
        }
    }
    r.color = glm::vec3(0,0,0);
    return false;
}

__global__ void RayIntersection(Ray* rays, int n, Ray* reflectedRays, Node* bvhhead, Isect* isectPoints, int* nw, int* ne, int* sw, int* se)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i+=stride)
    {
        glm::vec3 direction = rays[i].direction;
        glm::vec3 position = rays[i].position;

        reflectedRays[i].i = rays[i].i;
        reflectedRays[i].j = rays[i].j;
        reflectedRays[i].color = rays[i].color;
        reflectedRays[i].raytype = 1;
        
        float minT = 1000000000;
        SceneObject intersectObj;
        glm::vec3 minTnormal;
        glm::vec3 minTintersection;
        bool intersect = false;
        bvhTraverse(position,direction,bvhhead,intersect,minT,intersectObj,minTnormal,minTintersection);
        if(intersect)
        {
            
            if(rays[i].raytype == 0)
            {
                isectPoints[i].color = glm::vec3(0,0,0);
                reflectedRays[i].surfaceReflectiveCoef = intersectObj.reflective;
                isectPoints[i].reflectionCoef = glm::vec3(1.0f,1.0f,1.0f);
                isectPoints[i].color = 0.2f * intersectObj.ambient;
            }
            else if(rays[i].raytype == 1){
                isectPoints[i].reflectionCoef = rays[i].surfaceReflectiveCoef;
                reflectedRays[i].surfaceReflectiveCoef = rays[i].surfaceReflectiveCoef*intersectObj.reflective;
            }
            
            minTintersection = minTintersection+minTnormal*0.7f;
            reflectedRays[i].position = minTintersection;
            reflectedRays[i].direction = glm::normalize(glm::reflect(rays[i].direction, minTnormal));
            
             
            isectPoints[i].isectPoint = minTintersection;
            isectPoints[i].incidentDirection = rays[i].direction;
            isectPoints[i].normal = minTnormal;
            isectPoints[i].i = rays[i].i;
            isectPoints[i].j = rays[i].j;
            
            isectPoints[i].diffuse = intersectObj.diffuse;
            isectPoints[i].ambient = intersectObj.ambient;
            isectPoints[i].shininess = intersectObj.shininess;
            isectPoints[i].specular = intersectObj.specular;
            isectPoints[i].reflective = intersectObj.reflective;

            
        }
        else
        {
            Isect point;
            point.color = isectPoints[i].color;
            wallIntersection(point,rays[i],reflectedRays[i],bvhhead);
            isectPoints[i] = point;
            reflectedRays[i].position = point.isectPoint;
            reflectedRays[i].direction = glm::normalize(glm::reflect(rays[i].direction, point.normal));
            
            isectPoints[i].i = rays[i].i;
            isectPoints[i].j = rays[i].j;

        }
        /*if(reflectedRays[i].direction[0] <= 0.0f && reflectedRays[i].direction[1] >= 0.0f)
        {
            atomicAdd(nw,1);
        }
        else if(reflectedRays[i].direction[0] >= 0.0f && reflectedRays[i].direction[1] >= 0.0f)
        {
            atomicAdd(ne,1);
        }
        else if(reflectedRays[i].direction[0] <= 0.0f && reflectedRays[i].direction[1] <= 0.0f)
        {
            atomicAdd(sw,1);
        }
        else if(reflectedRays[i].direction[0] >= 0.0f && reflectedRays[i].direction[1] <= 0.0f)
        {
            atomicAdd(se,1);
        }*/
    
    }
}

__global__ void Shade(Isect* isectPoints, int n, Light* lights, int numlights, Node* bvhhead)
{
    float RAY_EPSILON = 0.000000001;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i+=stride)
    {
        glm::vec3 color;
        glm::vec3 direction = glm::normalize(isectPoints[i].incidentDirection);
        glm::vec3 intersection = isectPoints[i].isectPoint;
        glm::vec3 normal = isectPoints[i].normal;
        float distance;
        glm::vec3 toLight;
        glm::vec3 reflectFromLight;
        for(int j =0; j<numlights; j++)
        {
            Light l = lights[j];
            /*if(l.area)
            {
                glm::vec3 avgcolor;
                int lightSamples = 30;
                for(int i = 0 ;i<lightSamples;i++)
                {
                    float radius = curand_uniform(threadIdx.x*blockIdx.x)*l.radius;
                    float theta = curand_uniform(threadIdx.x*blockIdx.x)*(2*PI);
                    
                    float x = radius * cos(theta);
                    float z = radius * sin(theta);
                    l.position[0] += x;
                    l.position[2] += z;
                    toLight = glm::normalize(l.position-intersection);
                    reflectFromLight = -toLight;
                    
                    distance = 1.0f;
                    float minT = 1000000000;
                    SceneObject intersectObj;
                    glm::vec3 minTnormal;
                    glm::vec3 minTintersection;
                    bool shadow = false;
                    intersection = intersection + .01f * normal;
                    bvhTraverse(intersection,toLight,bvhhead,shadow,minT,intersectObj,minTnormal,minTintersection);
                    if(shadow)
                    {
                        if(minT>RAY_EPSILON)
                        {
                            glm::vec3 ipoint = intersection+minT*toLight;
                            float dtoLight = sqrt(pow(intersection[0]-l.position[0],2)+pow(intersection[1]-l.position[1],2)+pow(intersection[2]-l.position[2],2));
                            float dtoLightIntersection = sqrt(pow(ipoint[0]-intersection[0],2)+pow(ipoint[1]-intersection[1],2)+pow(ipoint[2]-intersection[2],2));
                            if(dtoLight>dtoLightIntersection)
                                distance = distance * 0;
                        }
                    }
                    
                    avgcolor += distance * l.color * ( .6f * isectPoints[i].diffuse * glm::max(glm::dot(toLight,normal),0.0f) + .2f * isectPoints[i].specular * glm::pow(glm::dot(glm::reflect(reflectFromLight, normal), -direction),isectPoints[i].shininess));
                }
                color += avgcolor/(float)lightSamples;
                
            }*/
            if(l.point)
            {
                float d = sqrt(pow(intersection[0]-l.position[0],2)+pow(intersection[1]-l.position[1],2)+pow(intersection[2]-l.position[2],2));
                distance = 1.0f/(l.constantTerm + l.linearTerm * d + l.quadraticTerm * pow(d,2));
                
                if(distance>1.5)
                    distance = .5;
                toLight = glm::normalize(l.position-intersection);
                reflectFromLight = -toLight;
                

                float minT = 1000000000;
                SceneObject intersectObj;
                glm::vec3 minTnormal;
                glm::vec3 minTintersection;
                bool shadow = false;
                intersection = intersection + .01f * normal;
                bvhTraverse(intersection,toLight,bvhhead,shadow,minT,intersectObj,minTnormal,minTintersection);
                
                if(shadow)
                {
                    glm::vec3 ipoint = intersection+minT*toLight;
                    float dtoLight = sqrt(pow(intersection[0]-l.position[0],2)+pow(intersection[1]-l.position[1],2)+pow(intersection[2]-l.position[2],2));
                    float dtoLightIntersection = sqrt(pow(ipoint[0]-intersection[0],2)+pow(ipoint[1]-intersection[1],2)+pow(ipoint[2]-intersection[2],2));
                    if(dtoLight>dtoLightIntersection)
                        distance = distance * 0;
                }
                color += distance * l.color * ( .6f * isectPoints[i].diffuse * glm::max(glm::dot(toLight,normal),0.0f) + .2f * isectPoints[i].specular * glm::pow(glm::dot(glm::reflect(reflectFromLight, normal), -direction),isectPoints[i].shininess));
            }
            else
            {
                distance = 1.0f;
                toLight = -glm::normalize(l.direction);
                reflectFromLight = glm::normalize(l.direction);

                float minT = 1000000000;
                SceneObject intersectObj;
                glm::vec3 minTnormal;
                glm::vec3 minTintersection;
                bool shadow = false;
                
		//check if light and surface normal are facing differnt directions
		float dotP = glm::dot(reflectFromLight,normal);
		if(dotP > -0.00001 )
                   shadow = true;
		else
		    bvhTraverse(intersection,toLight,bvhhead,shadow,minT,intersectObj,minTnormal,minTintersection);

                if(shadow)
                {
                    if(minT>RAY_EPSILON)
                        distance = distance * 0;
                }
                color += distance * l.color * ( .6f * isectPoints[i].diffuse * glm::max(glm::dot(toLight,normal),0.0f) + .2f * isectPoints[i].specular * glm::pow(glm::dot(glm::reflect(reflectFromLight, normal), -direction),isectPoints[i].shininess));
            }
        }
        isectPoints[i].color += isectPoints[i].reflectionCoef * color;
        
    }
}

void startRayTracing(float width, float height, unsigned char*& pixelcolorBuffer,glm::vec3 cameraPosition, glm::vec3 cameraDirection, std::vector<SceneObject*>& scene, std::vector<Light>& lights, Node* rootnode)
{

    //set the stack size for threads
    size_t limit = 4096;
    cudaDeviceSetLimit(cudaLimitStackSize,limit);
    

    Light* scenelights;
    int numlights = (int)lights.size();
    cudaMallocManaged(&scenelights,numlights*sizeof(Light));
    for(int i = 0; i<numlights; i++)
    {
        scenelights[i] = lights[i];
    }

    int totalRaysInSystem = width*height;
    
    //for primary ray calcuations
    glm::vec3 n = glm::normalize(cameraPosition-cameraDirection);
    glm::vec3 u = glm::normalize(glm::cross(glm::vec3(0,1,0),n));
    glm::vec3 v = glm::cross(n,u);
    float fov = 45/(180.0 / PI);
    float d = (height/tan(fov/2))/2;
    glm::vec3 L = (cameraPosition-n*d) - u * (width/2) - v*(height/2);
    
    //generate primary rays
    Ray *cudarays;
    cudaMallocManaged(&cudarays,totalRaysInSystem*sizeof(Ray));
    int blockSize = 256;
    int numBlocks = (totalRaysInSystem + blockSize -1)/blockSize;
    
    GeneratePrimaryRays<<<numBlocks,blockSize>>>(cudarays,totalRaysInSystem, L, u, v, cameraPosition);
    cudaDeviceSynchronize();
    
    Ray *reflectedRays;
    cudaMallocManaged(&reflectedRays,totalRaysInSystem*sizeof(Ray));
    
    Isect *cpuisectPoints = (Isect *)malloc(totalRaysInSystem*sizeof(Isect));;
    Isect *isectPoints;
    cudaMallocManaged(&isectPoints,totalRaysInSystem*sizeof(Isect));
    
    /*USED FOR RAY SORTING NOT IMPLEMENTED YET*/
    int *nw,*ne,*sw,*se;
    cudaMallocManaged(&nw,sizeof(int));
    cudaMallocManaged(&ne,sizeof(int));
    cudaMallocManaged(&sw,sizeof(int));
    cudaMallocManaged(&se,sizeof(int));
    //******************************************

    for(int i =0; i < numBounces; i++)
    {
        RayIntersection<<<numBlocks,blockSize>>>(cudarays,totalRaysInSystem,reflectedRays,rootnode,isectPoints,nw,ne,sw,se);
        cudaDeviceSynchronize();
        Shade<<<numBlocks, blockSize>>>(isectPoints,totalRaysInSystem,scenelights,numlights,rootnode);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuisectPoints, isectPoints, totalRaysInSystem*sizeof(Isect),cudaMemcpyDeviceToHost);
        cudaMemcpy(cudarays, reflectedRays, totalRaysInSystem*sizeof(Ray),cudaMemcpyDeviceToDevice);
    }
    for(int i = 0; i<totalRaysInSystem;i++)
    {
	cpuisectPoints[i].color = cpuisectPoints[i].color*255.0f;
        cpuisectPoints[i].color = glm::clamp(cpuisectPoints[i].color,glm::vec3(0.0f,0.0f,0.0f),glm::vec3(255.0f,255.0f,255.0f));
        pixelcolorBuffer[i*3] = cpuisectPoints[i].color[2];
	pixelcolorBuffer[i*3+1] = cpuisectPoints[i].color[1];
	pixelcolorBuffer[i*3+2] = cpuisectPoints[i].color[0];
    }
    cudaFree(cudarays);
    cudaFree(reflectedRays);
    cudaFree(isectPoints);
    free(cpuisectPoints);
    cudaFree(nw);
    cudaFree(ne);
    cudaFree(sw);
    cudaFree(se);
    cudaFree(scenelights);
    
}

