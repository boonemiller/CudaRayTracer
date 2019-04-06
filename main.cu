//
//  main.cpp
//  RayTracer
//
//  Created by Bo Miller on 1/2/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//

#include <stdio.h>
#include <vector>
#include <iostream>
#include "glm/glm/glm.hpp"
#include "SceneObjects.hpp"
#include "Ray.hpp"
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <chrono>
#include "bvh.hpp"
#include "objloader.hpp"

int main(int argc, const char * argv[]) {
    //*******Manually describing scene************
    float width = 720;
    float height = 360;
    std::vector<SceneObject> scene;
    
    std::vector< glm::vec3 > vertices;
    std::vector< glm::vec2 > uvs;
    std::vector< glm::vec3 > normals; // Won't be used at the moment.
    bool res = loadOBJ("teapot3.obj", vertices, uvs, normals);
    for(int i = 0;i<3072;i+=3)
    {
        SceneObject s1;
        s1.objNum = i;
        s1.radius = 0;
        s1.ambient = glm::vec3(0,0,1);
        s1.specular = glm::vec3(0.9,0.4,0);
        s1.diffuse = glm::vec3(0.8,0.3,0.1);
        s1.center = glm::vec3(0,0,0);
        s1.position = s1.center;
        
        s1.v1 = vertices[i];
        s1.v1Norm = glm::normalize(normals[i]);
        s1.v2 = vertices[i+1];
        s1.v2Norm = glm::normalize(normals[i+1]);
        s1.v3 = vertices[i+2];
        s1.v3Norm = glm::normalize(normals[i+2]);
        s1.shininess = 64;
        s1.reflective = glm::vec3(.5,.5,.5);
        s1.triangle = true;
        s1.sphere = false;
        
        scene.push_back(s1);
    }

    /*SceneObject s1;
    s1.objNum = 1;
    s1.radius = 0;
    s1.ambient = glm::vec3(0,0,1);
    s1.specular = glm::vec3(0.9,0.4,0);
    s1.diffuse = glm::vec3(0.8,0.3,0.1);
    s1.center = glm::vec3(0,0,0);
    s1.position = s1.center;
    
    s1.v1 = glm::vec3(-2,0,0);
    s1.v1Norm = glm::vec3(0,0,1);
    s1.v2 = glm::vec3(0,2,0);
    s1.v2Norm =  glm::vec3(0,0,1);
    s1.v3 = glm::vec3(2,0,0);
    s1.v3Norm =  glm::vec3(0,0,1);
    //printf("%f %f %f\n",s1.v1Norm[0],s1.v1Norm[1],s1.v1Norm[2]);
    s1.shininess = 64;
    s1.reflective = glm::vec3(.5,.5,.5);
    s1.triangle = true;
    s1.sphere = false;
    scene.push_back(s1)
    
    
    /*SceneObject s1;
    s1.objNum = 0;
    s1.radius = 1.0;
    s1.ambient = glm::vec3(0,0,1);
    s1.specular = glm::vec3(0.9,0.4,0);
    s1.diffuse = glm::vec3(0.8,0.3,0.1);
    s1.center = glm::vec3(-4,3,0);
    s1.position = s1.center;
    s1.shininess = 64;
    s1.reflective = glm::vec3(.5,.5,.5);
    s1.sphere = true;
    scene.push_back(s1);
    
    SceneObject s2;
    s2.objNum = 1;
    s2.radius = 1.5;
    s2.ambient = glm::vec3(1.0,1,1);
    s2.specular = glm::vec3(.5,.5,.5);
    s2.diffuse = glm::vec3(1,1,1);
    s2.center = glm::vec3(-2,0,0);
    s2.position = s2.center;
    s2.reflective = glm::vec3(.5,.5,.5);
    s2.shininess = 64;
    s2.sphere = true;
    scene.push_back(s2);
    
    SceneObject s3;
    s3.objNum = 2;
    s3.radius = 1;
    s3.ambient = glm::vec3(1.0,1.0,0.0);
    s3.specular = glm::vec3(.5,.5,.5);
    s3.diffuse = glm::vec3(1,1,1);
    s3.center = glm::vec3(1,0,3);
    s3.position = s3.center;
    s3.reflective = glm::vec3(.5,.5,.5);
    s3.shininess = 64;
    s3.sphere = true;
    scene.push_back(s3);
    
    SceneObject s4;
    s4.objNum = 3;
    s4.radius = 1;
    s4.ambient = glm::vec3(1.0,1.0,0.0);
    s4.specular = glm::vec3(.5,.5,.5);
    s4.diffuse = glm::vec3(1,1,1);
    s4.center = glm::vec3(2,0,3);
    s4.position = s4.center;
    s4.reflective = glm::vec3(.5,.5,.5);
    s4.shininess = 64;
    s4.sphere = true;
    scene.push_back(s4);
    
    SceneObject s5;
    s5.objNum = 4;
    s5.radius = .75;
    s5.ambient = glm::vec3(1.0,1.0,0.0);
    s5.specular = glm::vec3(.5,.5,.5);
    s5.diffuse = glm::vec3(1,1,1);
    s5.center = glm::vec3(1.5,1,3);
    s5.position = s5.center;
    s5.reflective = glm::vec3(.5,.5,.5);
    s5.shininess = 64;
    s5.sphere = true;
    scene.push_back(s5);
    
    SceneObject s6;
    s6.objNum = 5;
    s6.radius = .75;
    s6.ambient = glm::vec3(1.0,1.0,0.0);
    s6.specular = glm::vec3(.5,.5,.5);
    s6.diffuse = glm::vec3(1,1,1);
    s6.center = glm::vec3(1.5,-1,3);
    s6.position = s6.center;
    s6.reflective = glm::vec3(.5,.5,.5);
    s6.shininess = 64;
    s6.sphere = true;
    scene.push_back(s6);
    
    SceneObject s7;
    s7.objNum = 6;
    s7.radius = 1.0;
    s7.ambient = glm::vec3(1.0,1,1);
    s7.specular = glm::vec3(.5,.5,.5);
    s7.diffuse = glm::vec3(1,1,1);
    s7.center = glm::vec3(0,2,0);
    s7.position = s7.center;
    s7.reflective = glm::vec3(.5,.5,.5);
    s7.shininess = 64;
    s7.sphere = true;
    scene.push_back(s7);
    
    SceneObject s8;
    s8.objNum = 7;
    s8.radius = 1;
    s8.ambient = glm::vec3(0,1,0);
    s8.specular = glm::vec3(.2,.2,.2);
    s8.diffuse = glm::vec3(1,1,1);
    s8.center = glm::vec3(3,3,4);
    s8.position = s8.center;
    s8.reflective = glm::vec3(.5,.5,.5);
    s8.shininess = 64;
    s8.sphere = true;
    scene.push_back(s8);
    
    SceneObject s9;
    s9.objNum = 8;
    s9.radius = .75;
    s9.ambient = glm::vec3(.9,.9,.9);
    s9.specular = glm::vec3(.2,.2,.2);
    s9.diffuse = glm::vec3(1,1,1);
    s9.center = glm::vec3(3,2,5.5);
    s9.position = s9.center;
    s9.reflective = glm::vec3(.5,.5,.5);
    s9.shininess = 64;
    s9.sphere = true;
    scene.push_back(s9);
    
    SceneObject s10;
    s10.objNum = 8;
    s10.radius = 1;
    s10.ambient = glm::vec3(0,0,1);
    s10.specular = glm::vec3(1,1,1);
    s10.diffuse = glm::vec3(1,1,1);
    s10.center = glm::vec3(-3,-.5,5);
    s10.position = s10.center;
    s10.reflective = glm::vec3(.5,.5,.5);
    s10.shininess = 64;
    s10.sphere = true;
    scene.push_back(s10);
    
    SceneObject s11;
    s11.objNum = 9;
    s11.radius = 1;
    s11.ambient = glm::vec3(1,0,1);
    s11.specular = glm::vec3(1,1,1);
    s11.diffuse = glm::vec3(1,1,1);
    s11.center = glm::vec3(-4.5,2,4);
    s11.position = s11.center;
    s11.reflective = glm::vec3(.5,.5,.5);
    s11.shininess = 64;
    s11.sphere = true;
    scene.push_back(s11);
    
    SceneObject s12;
    s12.objNum = 11;
    s12.radius = 1;
    s12.ambient = glm::vec3(0,1,1);
    s12.specular = glm::vec3(.5,.5,.5);
    s12.diffuse = glm::vec3(1,1,1);
    s12.center = glm::vec3(2.5,3.5,-3.5);
    s12.position = s12.center;
    s12.reflective = glm::vec3(.3,.3,.3);
    s12.shininess = 64;
    s12.sphere = true;
    scene.push_back(s12);*/
    
    glm::vec3 cameraDirection = glm::vec3(0,0,0);
    //************************************
    
    
    //**************Adding Lights*****************
    std::vector<Light> lights;
    Light l1;
    l1.direction = glm::vec3(2, -1, -1);
    l1.color = glm::vec3(1.0, 1.0, 1.0);
    l1.point = false;
    l1.area = false;
    lights.push_back(l1);
    
    Light l2;
    l2.point = true;
    l2.area = false;
    l2.position = glm::vec3(4, 3, -4);
    l2.color = glm::vec3(.5, .5, .5);
    l2.constantTerm = 0.25f;
    l2.linearTerm = 0.003372407f;
    l2.quadraticTerm = 0.000045492f;
    //lights.push_back(l2);
    
    
    Light l3;
    l3.direction = glm::vec3(-2,-1,1);
    l3.color = glm::vec3(1,1,1);
    l3.point = false;
    l3.area = false;
    //lights.push_back(l3);
    
    //area light
    /*Light l4;
    l4.position = glm::vec3(0,8,0);
    l4.area = true;
    l4.point = false;
    l4.color = glm::vec3(1,1,1);
    l4.radius = 1;
    lights.push_back(l4);*/
    //********************************************
    
    
    //*****Builds bvh tree*****

    Node* cudaRoot;
    cudaMallocManaged(&cudaRoot,sizeof(Node));
    cudaRoot->isleaf = false;
    cudaRoot->maxX = std::numeric_limits<float>::min();
    cudaRoot->minX = std::numeric_limits<float>::max();
    cudaRoot->maxY = std::numeric_limits<float>::min();
    cudaRoot->minY = std::numeric_limits<float>::max();;
    cudaRoot->maxZ = std::numeric_limits<float>::min();
    cudaRoot->minZ = std::numeric_limits<float>::max();;
    
    //creates world bounding box information, includes all the objects
    for(SceneObject obj: scene)
    {
        if(obj.sphere)
        {
            if(obj.position[0]-obj.radius < cudaRoot->minX)
                cudaRoot->minX = obj.position[0]-obj.radius;
            if(obj.position[1]-obj.radius < cudaRoot->minY)
                cudaRoot->minY = obj.position[1]-obj.radius;
            if(obj.position[2]-obj.radius < cudaRoot->minZ)
                cudaRoot->minZ = obj.position[2]-obj.radius;
        
            if(obj.position[0]+obj.radius > cudaRoot->maxX)
                cudaRoot->maxX = obj.position[0]+obj.radius;
            if(obj.position[1]+obj.radius > cudaRoot->maxY)
                cudaRoot->maxY = obj.position[1]+obj.radius;
            if(obj.position[2]+obj.radius > cudaRoot->maxZ)
                cudaRoot->maxZ = obj.position[2]+obj.radius;
        }
        else if(obj.triangle)
        {
            if(obj.v1[0] < cudaRoot->minX)
                cudaRoot->minX = obj.v1[0];
            if(obj.v1[1] < cudaRoot->minY)
                cudaRoot->minY = obj.v1[1];
            if(obj.v1[2] < cudaRoot->minZ)
                cudaRoot->minZ = obj.v1[2];
            
            if(obj.v1[0] > cudaRoot->maxX)
                cudaRoot->maxX = obj.v1[0];
            if(obj.v1[1] > cudaRoot->maxY)
                cudaRoot->maxY = obj.v1[1];
            if(obj.v1[2] > cudaRoot->maxZ)
                cudaRoot->maxZ = obj.v1[2];
            
            if(obj.v2[0] < cudaRoot->minX)
                cudaRoot->minX = obj.v2[0];
            if(obj.v2[1] < cudaRoot->minY)
                cudaRoot->minY = obj.v2[1];
            if(obj.v2[2] < cudaRoot->minZ)
                cudaRoot->minZ = obj.v2[2];
            
            if(obj.v2[0] > cudaRoot->maxX)
                cudaRoot->maxX = obj.v2[0];
            if(obj.v2[1] > cudaRoot->maxY)
                cudaRoot->maxY = obj.v2[1];
            if(obj.v2[2] > cudaRoot->maxZ)
                cudaRoot->maxZ = obj.v2[2];
            
            if(obj.v3[0] < cudaRoot->minX)
                cudaRoot->minX = obj.v3[0];
            if(obj.v3[1] < cudaRoot->minY)
                cudaRoot->minY = obj.v3[1];
            if(obj.v3[2] < cudaRoot->minZ)
                cudaRoot->minZ = obj.v3[2];
            
            if(obj.v3[0] > cudaRoot->maxX)
                cudaRoot->maxX = obj.v3[0];
            if(obj.v3[1] > cudaRoot->maxY)
                cudaRoot->maxY = obj.v3[1];
            if(obj.v3[2] > cudaRoot->maxZ)
                cudaRoot->maxZ = obj.v3[2];
        }
        
    }
    
    if(cudaRoot->maxX-cudaRoot->minX > cudaRoot->maxY-cudaRoot->minY)
    {
        if(cudaRoot->maxX-cudaRoot->minX > cudaRoot->maxZ - cudaRoot->minZ)
        {
            cudaRoot->longestAxis = 0;
            cudaRoot->midpoint = (cudaRoot->maxX+cudaRoot->minX)/2;
        }
    }
    if(cudaRoot->maxY-cudaRoot->minY > cudaRoot->maxX-cudaRoot->minX)
    {
        if(cudaRoot->maxY-cudaRoot->minY > cudaRoot->maxZ-cudaRoot->minZ)
        {
            cudaRoot->longestAxis = 1;
            cudaRoot->midpoint = (cudaRoot->maxY+cudaRoot->minY)/2;
        }
    }
    if(cudaRoot->maxZ-cudaRoot->minZ > cudaRoot->maxX-cudaRoot->minX)
    {
        if(cudaRoot->maxZ-cudaRoot->minZ > cudaRoot->maxY-cudaRoot->minY)
        {
            cudaRoot->longestAxis = 2;
            cudaRoot->midpoint = (cudaRoot->maxZ+cudaRoot->minZ)/2;
        }
    }
    glm::vec3 cameraPosition = glm::vec3(0,cudaRoot->maxY+20,cudaRoot->maxZ+200);
    constructTree(scene, cudaRoot);
    
    
    
    /*Writes pixel buffer to a .bmp file*/
    {
        int w = (int) width;
        int h = (int) height;
        
        /*Declare pixel buffer and start ray tracing*/
        uint8_t *pix;
        pix = (uint8_t *)malloc(w*h*3 * sizeof(uint8_t));
        memset(pix, 0xff, w*h*3);
        float t = 0;
        float buffer[360][720][3];
        //auto start = std::chrono::high_resolution_clock::now();
        startRayTracing(width, height, buffer, cameraPosition, cameraDirection, scene, lights, cudaRoot);
        //auto stop = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //t += duration.count();
        
        for(int i = 0; i<h; i++)
        {
            for(int j = 0;j<w;j++)
            {
                int elem = (i*w*3) + (j*3);
                pix[elem+0] = (uint8_t) (buffer[i][j][0] * 255.0);
                pix[elem+1] = (uint8_t) (buffer[i][j][1] * 255.0);
                pix[elem+2] = (uint8_t) (buffer[i][j][2] * 255.0);
                
            }
        }
        /*Denoising loop*/
        t = 0;
        for(int ii = 0; ii<20; ii++)
        {
            
            float buffer1[360][720][3];
            auto start = std::chrono::high_resolution_clock::now();
            startRayTracing(width, height, buffer1, cameraPosition, cameraDirection, scene, lights, cudaRoot);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            t += duration.count();
            
            for(int i = 0; i<h; i++)
            {
                for(int j = 0;j<w;j++)
                {
                    int elem = (i*w*3) + (j*3);
                    pix[elem+0] = std::max(pix[elem+0], (uint8_t) (buffer1[i][j][0] * 255.0));
                    pix[elem+1] = std::max(pix[elem+1], (uint8_t) (buffer1[i][j][1] * 255.0));
                    pix[elem+2] = std::max(pix[elem+2], (uint8_t) (buffer1[i][j][2] * 255.0));
                    buffer1[i][j][0]=0.0;
                    buffer1[i][j][1]=0.0;
                    buffer1[i][j][2]=0.0;
                }
            }
            
            
        }
        t = t/20;
        std::cout << "AVG Trace Time: "
        << t/1000000 << " seconds\n" << std::endl;
        stbi_write_bmp("./fb.bmp", w, h, 3, pix);
    }
    
    return 0;
}
