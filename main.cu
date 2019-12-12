//
//  main.cpp
//  RayTracer
//
//  Created by Bo Miller on 1/2/19.
//  Copyright © 2019 Bo Miller. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <iostream>
#include "glm/glm/glm.hpp"
#include "glm/glm/gtc/matrix_transform.hpp"
#include "SceneObjects.hpp"
#include "Ray.hpp"
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <chrono>
#include "bvh.hpp"
#include "objloader.hpp"
//#include <GL/freeglut.h>
//#include <GL/gl.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
GLFWwindow* window;
#include "Shader.hpp"



void newBuild(Node*& cudaRoot, std::deque<Node *>& leafs,std::vector<SceneObject *>& scene){
    cudaRoot->isleaf = false;
    cudaRoot->maxX = std::numeric_limits<float>::min();
    cudaRoot->minX = std::numeric_limits<float>::max();
    cudaRoot->maxY = std::numeric_limits<float>::min();
    cudaRoot->minY = std::numeric_limits<float>::max();;
    cudaRoot->maxZ = std::numeric_limits<float>::min();
    cudaRoot->minZ = std::numeric_limits<float>::max();;
    
    //creates world bounding box information, includes all the objects
    for(SceneObject* obj: scene)
    {
        if(obj->sphere)
        {
            if(obj->position[0]-obj->radius < cudaRoot->minX)
                cudaRoot->minX = obj->position[0]-obj->radius;
            if(obj->position[1]-obj->radius < cudaRoot->minY)
                cudaRoot->minY = obj->position[1]-obj->radius;
            if(obj->position[2]-obj->radius < cudaRoot->minZ)
                cudaRoot->minZ = obj->position[2]-obj->radius;
        
            if(obj->position[0]+obj->radius > cudaRoot->maxX)
                cudaRoot->maxX = obj->position[0]+obj->radius;
            if(obj->position[1]+obj->radius > cudaRoot->maxY)
                cudaRoot->maxY = obj->position[1]+obj->radius;
            if(obj->position[2]+obj->radius > cudaRoot->maxZ)
                cudaRoot->maxZ = obj->position[2]+obj->radius;
        }
        else if(obj->triangle)
        {
            if(obj->v1[0] < cudaRoot->minX)
                cudaRoot->minX = obj->v1[0];
            if(obj->v1[1] < cudaRoot->minY)
                cudaRoot->minY = obj->v1[1];
            if(obj->v1[2] < cudaRoot->minZ)
                cudaRoot->minZ = obj->v1[2];
            
            if(obj->v1[0] > cudaRoot->maxX)
                cudaRoot->maxX = obj->v1[0];
            if(obj->v1[1] > cudaRoot->maxY)
                cudaRoot->maxY = obj->v1[1];
            if(obj->v1[2] > cudaRoot->maxZ)
                cudaRoot->maxZ = obj->v1[2];
            
            if(obj->v2[0] < cudaRoot->minX)
                cudaRoot->minX = obj->v2[0];
            if(obj->v2[1] < cudaRoot->minY)
                cudaRoot->minY = obj->v2[1];
            if(obj->v2[2] < cudaRoot->minZ)
                cudaRoot->minZ = obj->v2[2];
            
            if(obj->v2[0] > cudaRoot->maxX)
                cudaRoot->maxX = obj->v2[0];
            if(obj->v2[1] > cudaRoot->maxY)
                cudaRoot->maxY = obj->v2[1];
            if(obj->v2[2] > cudaRoot->maxZ)
                cudaRoot->maxZ = obj->v2[2];
            
            if(obj->v3[0] < cudaRoot->minX)
                cudaRoot->minX = obj->v3[0];
            if(obj->v3[1] < cudaRoot->minY)
                cudaRoot->minY = obj->v3[1];
            if(obj->v3[2] < cudaRoot->minZ)
                cudaRoot->minZ = obj->v3[2];
            
            if(obj->v3[0] > cudaRoot->maxX)
                cudaRoot->maxX = obj->v3[0];
            if(obj->v3[1] > cudaRoot->maxY)
                cudaRoot->maxY = obj->v3[1];
            if(obj->v3[2] > cudaRoot->maxZ)
                cudaRoot->maxZ = obj->v3[2];
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
    cudaRoot->parent = NULL;
    cudaRoot->nodeNum = 0;
    //printf("INSIDE HEREH\n");
    constructTree(scene, cudaRoot, leafs, cudaRoot);
}



int main(int argc, char * argv[]) {
 	float width = 720;
        float height = 360;
	// Initialise GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( width, height, "GPU Ray Tracer", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS); 

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( "VertexShader.vertexshader", "FragmentShader.fragmentshader" );


	
	//GLuint Texture = loadDDS("uvtemplate.DDS");
	
	// Get a handle for our "myTextureSampler" uniform
	


	static const GLfloat g_vertex_buffer_data[] = { 
		-1.0f,-1.0f,0.0f,
		-1.0f,1.0f,0.0f,
		1.0f,-1.0f,0.0f,
		-1.0f,1.0f,0.0f,
		1.0f,1.0f,0.0f,
		1.0f,-1.0f,0.0f
	};

	// Two UV coordinatesfor each vertex. They were created with Blender.
	static const GLfloat g_uv_buffer_data[] = {
		0.0f,0.0f,
		0.0f,1.0f,
		1.0f,0.0f,
		0.0f,1.0f,
		1.0f,1.0f,
		1.0f,0.0f 
		
	};

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

    //*******Manually describing scene************
    
    std::vector<SceneObject *> scene;
    
    std::vector< glm::vec3 > vertices;
    std::vector< glm::vec2 > uvs;
    std::vector< glm::vec3 > normals; // Won't be used at the moment.
    bool res = loadOBJ("teapot3.obj", vertices, uvs, normals);
    for(int i = 0;i<3072;i+=3)
    {
	SceneObject* s1;
        cudaMallocManaged(&s1,sizeof(SceneObject));
        s1->objNum = i;
        s1->radius = 0;
        s1->ambient = glm::vec3(0,0,1);
        s1->specular = glm::vec3(0.9,0.4,0);
        s1->diffuse = glm::vec3(0.8,0.3,0.1);
        s1->center = glm::vec3(0,0,0);
        s1->position = s1->center;
        
        s1->v1 = vertices[i];
        s1->v1Norm = glm::normalize(normals[i]);
        s1->v2 = vertices[i+1];
        s1->v2Norm = glm::normalize(normals[i+1]);
        s1->v3 = vertices[i+2];
        s1->v3Norm = glm::normalize(normals[i+2]);
        s1->shininess = 64;
        s1->reflective = glm::vec3(.5,.5,.5);
        s1->triangle = true;
        s1->sphere = false;
        
        scene.push_back(s1);
    }

    /*SceneObject* s1;
    s1->objNum = 1;
    s1->radius = 0;
    s1->ambient = glm::vec3(0,0,1);
    s1->specular = glm::vec3(0.9,0.4,0);
    s1->diffuse = glm::vec3(0.8,0.3,0.1);
    s1->center = glm::vec3(0,0,0);
    s1->position = s1.center;
    
    s1->v1 = glm::vec3(-2,0,0);
    s1->v1Norm = glm::vec3(0,0,1);
    s1->v2 = glm::vec3(0,2,0);
    s1->v2Norm =  glm::vec3(0,0,1);
    s1->v3 = glm::vec3(2,0,0);
    s1->v3Norm =  glm::vec3(0,0,1);
    //printf("%f %f %f\n",s1.v1Norm[0],s1.v1Norm[1],s1.v1Norm[2]);
    s1.shininess = 64;
    s1.reflective = glm::vec3(.5,.5,.5);
    s1.triangle = true;
    s1.sphere = false;
    scene.push_back(s1);*/
    
    
    /*SceneObject* s1;
    cudaMallocManaged(&s1,sizeof(SceneObject));
    float r1 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    float r2 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    float r3 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    s1->direction = glm::normalize(glm::vec3(r1,r2,r3));
    s1->speed = .01 + ((float)rand()/(float)(RAND_MAX) *.3);
    s1->objNum = 0;
    s1->radius = 2.0;
    s1->ambient = glm::vec3(0,0,1);
    s1->specular = glm::vec3(0.9,0.4,0);
    s1->diffuse = glm::vec3(0.8,0.3,0.1);
    s1->center = glm::vec3(-4,3,0);
    s1->position = s1->center;
    s1->shininess = 64;
    s1->reflective = glm::vec3(.5,.5,.5);
    s1->sphere = true;
    scene.push_back(s1);*/
    
    SceneObject* s2;
    cudaMallocManaged(&s2,sizeof(SceneObject));
    float r1 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    float r2 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    float r3 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    s2->direction = glm::vec3(r1,r2,r3);
    s2->speed = .01 + ((float)rand()/(float)(RAND_MAX) *.3);
    s2->objNum = 1;
    s2->radius = 20;
    s2->ambient = glm::vec3(1.0,1,1);
    s2->specular = glm::vec3(.5,.5,.5);
    s2->diffuse = glm::vec3(1,1,1);
    s2->center = glm::vec3(-40,40,0);
    s2->position = s2->center;
    s2->reflective = glm::vec3(.5,.5,.5);
    s2->shininess = 64;
    s2->sphere = true;
    scene.push_back(s2);
    
    /*SceneObject* s3;
    cudaMallocManaged(&s3,sizeof(SceneObject));
    r1 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    r2 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    r3 = -1 + ((float)rand()/(float)(RAND_MAX) * 2);
    s3->direction = glm::vec3(r1,r2,r3);
    s3->speed = .01 + ((float)rand()/(float)(RAND_MAX) *.3);
    s3->objNum = 2;
    s3->radius = 2;
    s3->ambient = glm::vec3(1.0,1.0,0.0);
    s3->specular = glm::vec3(.5,.5,.5);
    s3->diffuse = glm::vec3(1,1,1);
    s3->center = glm::vec3(1,0,3);
    s3->position = s3->center;
    s3->reflective = glm::vec3(.5,.5,.5);
    s3->shininess = 64;
    s3->sphere = true;
    scene.push_back(s3);
    
    /*SceneObject s4;
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
    lights.push_back(l3);
    
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
    std::deque<Node *> leafs;
    newBuild(cudaRoot,leafs,scene);

    glm::vec3 cameraPosition = glm::vec3(0,50,250);
   
    
    /*Writes pixel buffer to a .bmp file*/
    unsigned char* buffer1;
    buffer1 = (unsigned char *)malloc(259200 * 3 * sizeof(unsigned char));
    uint8_t *pix;
    pix = (uint8_t *)malloc(width*height*3 * sizeof(uint8_t));
    memset(pix, 0xff, width*height*3);
    int count = 0;
    int direction = 1;
    double lastTime = glfwGetTime();
    int nbFrames = 0;
    int curFrame = 0;
    do{
        // Measure speed
        double currentTime = glfwGetTime();
        nbFrames++;
	curFrame++;
        if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
          // printf and reset timer
          printf("%f ms/frame %f FPS\n", 1000.0/double(nbFrames), 1000.0f/(1000.0/double(nbFrames)));
          nbFrames = 0;
          lastTime += 1.0;
        }
	count++;
	if(curFrame<20)        
	   refitTree(leafs);
	else{
	   freeTree(cudaRoot);
	   leafs.clear();
	   cudaMallocManaged(&cudaRoot,sizeof(Node));
	   newBuild(cudaRoot,leafs,scene);
	}
	startRayTracing(width, height, buffer1, cameraPosition, cameraDirection, scene, lights, cudaRoot);

            
	for(SceneObject* s : scene)
	{
	    if(s->triangle)
	    {
	      glm::mat4 trans = glm::mat4(1.0);
	      trans = glm::rotate(trans, 0.01f, glm::vec3(0.0f,1.0f,0.0f));
	      s->v1 = glm::vec3(trans * glm::vec4(s->v1,1.0));
	      s->v2 = glm::vec3(trans * glm::vec4(s->v2,1.0));
	      s->v3 = glm::vec3(trans * glm::vec4(s->v3,1.0));
	      s->v1Norm = glm::vec3(trans * glm::vec4(s->v1Norm,1.0));
	      s->v2Norm = glm::vec3(trans * glm::vec4(s->v2Norm,1.0));
	      s->v3Norm = glm::vec3(trans * glm::vec4(s->v3Norm,1.0));
	    }
	    if(s->sphere){
	      if(s->position[0]+s->radius > 200.0f)
		  s->direction = glm::reflect(s->direction,glm::vec3(-1,0,0));
	      if(s->position[0]-s->radius < -200.0f)
		  s->direction = glm::reflect(s->direction,glm::vec3(1,0,0));
              if(s->position[1]+s->radius > 200.0f)
		  s->direction = glm::reflect(s->direction,glm::vec3(0,-1,0));
              if(s->position[1]-s->radius < -200.0f)
		  s->direction = glm::reflect(s->direction,glm::vec3(0,1,0));
              if(s->position[2]+s->radius > 300.0f)
		  s->direction = glm::reflect(s->direction,glm::vec3(0,0,-1));
              if(s->position[2]-s->radius < -300.0f)
		  s->direction = glm::reflect(s->direction,glm::vec3(0,0,1));
	      s->position += s->direction * s->speed * 8.0f;
	    }
	}
        GLuint TextureIDs  = glGetUniformLocation(programID, "myTextureSampler");
        // Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use our shader
	glUseProgram(programID);


	// Bind our texture in Texture Unit 0
	GLuint textureID;
		
	glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, buffer1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	// ... which requires mipmaps. Generate them automatically.
	glGenerateMipmap(GL_TEXTURE_2D);
		
	glActiveTexture(GL_TEXTURE0);

	// Set our "myTextureSampler" sampler to use Texture Unit 0
	glUniform1i(TextureIDs, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		2,                                // size : U+V => 2
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// Draw the triangle !
	glDrawArrays(GL_TRIANGLES, 0, 2*3); // 12*3 indices starting at 0 -> 12 triangles

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );

	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteProgram(programID);
	//glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;

}
