/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

/// Features
#define BLOOM 1
#define BLOOM2PASS 0
#define BILINEAR 1
#define USE_TEXTURES 1

/// Constant Settings
#define GAMMA 2.2f
#define EXPOSURE 1.0f

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;
		glm::vec3 eyePos;
		glm::vec3 eyeNor;
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth, texHeight;
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;
		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex;
		int texWidth, texHeight;
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

#if BLOOM

static glm::vec3 *dev_bloom1 = NULL;
static glm::vec3 *dev_bloom2 = NULL;

#endif

static int * dev_depth = NULL;
static int * dev_fragMutex = NULL;


/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__
void toneMap(const int w, const int h, glm::vec3 *framebuffer, const float gamma, const float exposure) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 col = framebuffer[index];
		//col = glm::pow(col, glm::vec3(1.0f / gamma));
		col = glm::vec3(1.0f) - glm::exp(-exposure * col);
		col = glm::pow(col, glm::vec3(1.0f / gamma));
		framebuffer[index] = col;
	}
}

__device__ __host__
glm::vec3 bytesToRGB(const TextureData* textureData, const int idx) {	
	return glm::vec3(textureData[idx] / 255.f, textureData[idx + 1] / 255.f, textureData[idx + 2] / 255.f);
}

// get a texture color, 
__device__ __host__
glm::vec3 texture2D(const int w, const int h, const TextureData* textureData, const glm::vec2 UV) {
	glm::vec2 uv = glm::mod(UV, glm::vec2(1.0f)); // repeat UV

	float xf = floor(uv.x * w);
	float yf = floor(uv.y * h);

	int x = (int)xf;
	int y = (int)yf;

	glm::vec3 col;


#if BILINEAR
	float xw = uv.x * w - xf;
	float yw = uv.y * h - yf;

	glm::vec3 col00, col01, col10, col11;
	col00 = bytesToRGB(textureData, 3 * (x + y * w));
	col01 = bytesToRGB(textureData, 3 * (x + 1 + y * w));
	col10 = bytesToRGB(textureData, 3 * (x + (y + 1) * w));
	col11 = bytesToRGB(textureData, 3 * (x + 1 + (y + 1) * w));

	col = yw * (xw * col00 + (1.f - xw) * col01) + (1.f - yw) * (xw * col10 + (1.f - xw) * col11);
#else 
	int idx = 3 * (x + y * w);
	col = bytesToRGB(textureData, idx);
#endif

	// apply gamma correction
	col = glm::pow(col, glm::vec3(GAMMA));

	return col;
}

#if BLOOM

// check for color components above 1, transfer to buffer with half res
__global__
void bloomHighPass(int wHalf, int hHalf, const glm::vec3 *framebuffer, glm::vec3 *bloombuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int bloomIdx = x + y * wHalf;

	if (x < wHalf && y < hHalf) {
		glm::vec3 col = glm::vec3(0);
		// get avg of 4 px from framebuffer
		for (int yOff = 0; yOff <= 1; yOff++) {
			for (int xOff = 0; xOff <= 1; xOff++) {
				int x2 = 2 * x + xOff;
				int y2 = 2 * y + yOff;

				int fbIdx = x2 + y2 * (2 * wHalf);
				glm::vec3 fbCol = framebuffer[fbIdx];

				float intensity = dot(fbCol, fbCol);
				intensity -= 3.f; // threshold
				//intensity *= 0.5f; // stretch response curve
				intensity = intensity < 0.f ? 0.f : intensity; // clamp
				intensity = intensity > 1.f ? 1.f : intensity;

				intensity = intensity * intensity * (3.f - 2.f * intensity); // smoothstep

				col += 0.25f * intensity * fbCol;
			}
		}
		bloombuffer[bloomIdx] = col;
	}
}

__global__
void bloomHorizontalGather(int w, int h, const glm::vec3 *bufIn, glm::vec3 *bufOut) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = x + y * w;

	if (x < w && y < h) {
		float weight[5] = { 0.227027027f, 0.194594595f, 0.121621622f, 0.054054054f, 0.016216216f};
		glm::vec3 col = bufIn[idx] * weight[0];
		for (int i = 1; i < 5; i++) {
			int prev = x - i;
			int next = x + i;
			prev = prev < 0 ? 0 : prev;
			next = next >= w ? w - 1 : next;

			col += weight[i] * bufIn[prev + y * w];
			col += weight[i] * bufIn[next + y * w];
		}

		bufOut[idx] = col;
	}
}

__global__ 
void bloomVerticalGather(int w, int h, const glm::vec3 *bufIn, glm::vec3 *bufOut) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = x + y * w;

	if (x < w && y < h) {
		float weight[5] = { 0.227027027f, 0.194594595f, 0.121621622f, 0.054054054f, 0.016216216f };
		glm::vec3 col = bufIn[idx] * weight[0];
		for (int i = 1; i < 5; i++) {
			int prev = y - i;
			int next = y + i;
			prev = prev < 0 ? 0 : prev;
			next = next >= h ? h - 1 : next;

			col += weight[i] * bufIn[x + prev * w];
			col += weight[i] * bufIn[x + next * w];
		}

		bufOut[idx] = col;
	}
}

__global__
void bloomComposite(int w, int h, glm::vec3 *framebuffer, const glm::vec3 *bloombuffer) {
	// going to bilinear upsample the bloomBuffer to get composite color
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = x + y * w;

	if (x < w && y < h) {
		// get 4 samples of bloom buffer and interpolate
		// if the current px is odd, it's in latter half of x / y of pixel
		float wx = x & 1 ? 0.75f : 0.25f;
		float wy = y & 1 ? 0.75f : 0.25f;

		int wb = w / 2;
		int hb = h / 2;
		int xb = x / 2;
		int yb = y / 2;

		// quadrant offset
		int x0 = x & 1 ? (xb) : (xb > 0 ? xb - 1 : 0);
		int x1 = x & 1 ? (xb >= (wb - 1) ? wb - 1 : xb + 1) : (xb);

		int y0 = y & 1 ? (yb) : (yb > 0 ? yb - 1 : 0);
		int y1 = y & 1 ? (yb >= (hb - 1) ? hb - 1 : yb + 1) : (yb);

		glm::vec3 col00, col01, col10, col11;

		col00 = bloombuffer[x0 + y0 * wb];
		col01 = bloombuffer[x1 + y0 * wb];
		col10 = bloombuffer[x0 + y1 * wb];
		col11 = bloombuffer[x1 + y1 * wb];

		// add the color, HDR is resolved by tone mapping
		framebuffer[idx] += wy * (wx * col00 + (1.f - wx) * col01) + (1.f - wy) * (wx * col10 + (1.f - wx) * col11);
	}
}

#endif

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		if (glm::length(fragmentBuffer[index].color) < 0.0001f) {
			framebuffer[index] = glm::vec3(0);
			return;
		}

		glm::vec3 lightDir[3] = { 
			glm::normalize(glm::vec3(1)),
			glm::normalize(glm::vec3(-1, -0.1, -0.8)),
			glm::normalize(glm::vec3(0, -1, 0)) 
		};

		float lightIntensity[3] = { 
			1.5f, 0.3f, 0.2f 
		};

		glm::vec3 lightCol[3] = {
			glm::vec3(1.0f, 0.9f, 0.7f),
			glm::vec3(0.8f, 0.9f, 1.0f),
			glm::vec3(0.4f, 1.0f, 0.5f)
		};

		glm::vec3 matDiffuse;
#if USE_TEXTURES
		if (fragmentBuffer[index].dev_diffuseTex != NULL) {
			matDiffuse = texture2D(fragmentBuffer[index].texWidth, fragmentBuffer[index].texHeight,
				fragmentBuffer[index].dev_diffuseTex, fragmentBuffer[index].texcoord0);
			matDiffuse = glm::max(matDiffuse, glm::vec3(0.05f));
		}
		else {
			matDiffuse = glm::vec3(0.75f);
		}
#else 
		matDiffuse = glm::vec3(0.75f);
#endif

		// simple blinn phong
		glm::vec3 col = glm::vec3(0);
		glm::vec3 nor = fragmentBuffer[index].eyeNor;

		for (int i = 0; i < 3; i++) {
			glm::vec3 halfVec = glm::normalize(lightDir[i] - glm::normalize(fragmentBuffer[index].eyePos));

			float lambert = glm::dot(nor, lightDir[i]);
			lambert = lambert < 0 ? 0 : lambert;
			float blinn = pow(glm::dot(halfVec, nor), 64.0f);
			blinn = blinn < 0 ? 0 : blinn;

			col += lightIntensity[i] * lightCol[i] * (glm::vec3(blinn) + matDiffuse * lambert);
		}

		framebuffer[index] = col;
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(dev_fragMutex);
	cudaMalloc(&dev_fragMutex, width * height * sizeof(int));

#if BLOOM
	cudaFree(dev_bloom1);
	cudaFree(dev_bloom2);
	cudaMalloc(&dev_bloom1, width * height / 4 * sizeof(glm::vec3));
	cudaMalloc(&dev_bloom2, width * height / 4 * sizeof(glm::vec3));
	cudaMemset(dev_bloom1, 0, width * height / 4 * sizeof(glm::vec3));
	cudaMemset(dev_bloom2, 0, width * height / 4 * sizeof(glm::vec3));
#endif

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


__global__ 
void initMutex(int w, int h, int * mutex) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		mutex[index] = 0;
	}
}

/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}
}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		glm::vec4 posIn = glm::vec4(primitive.dev_position[vid], 1.0f);
		
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		glm::vec4 posTransformed = MVP * posIn;
		// divide the pos by its w element to transform into NDC space
		posTransformed /= posTransformed.w;
		// Finally transform x and y to viewport space
		posTransformed.x = 0.5f * (posTransformed.x + 1.0f) * width;
		posTransformed.y = 0.5f * (-posTransformed.y + 1.0f) * height;

		primitive.dev_verticesOut[vid].pos = posTransformed; // screen position
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * posIn); // view position for lighting

#if USE_TEXTURES
		if (primitive.dev_diffuseTex != NULL) {
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
			primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
			primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
		}
		else {
			primitive.dev_verticesOut[vid].dev_diffuseTex = NULL;
		}
#endif
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {
		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
	}
}
// parallelize rasterization by triangle
__global__ void _rasterizeTriangle(const int numTris, const Primitive* primitives, 
	Fragment* frags, int* depthBuffer, const int width, const int height, int * mutex) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numTris) return;

	Primitive pri = primitives[idx];
	glm::vec3 tri[3] = { glm::vec3(pri.v[0].pos), glm::vec3(pri.v[1].pos), glm::vec3(pri.v[2].pos) };
	glm::vec3 triNor[3] = { glm::vec3(pri.v[0].eyeNor), glm::vec3(pri.v[1].eyeNor), glm::vec3(pri.v[2].eyeNor) };
	glm::vec3 triPos[3] = { glm::vec3(pri.v[0].eyePos), glm::vec3(pri.v[1].eyePos), glm::vec3(pri.v[2].eyePos) };
	AABB aabb = getAABBForTriangle(tri);
	
	for (int y = (int) aabb.min.y; y <= (int) aabb.max.y; y++) {
		if (y < 0 || y > height) continue;
		for (int x = (int) aabb.min.x; x <= (int) aabb.max.x; x++) {
			if (x < 0 || x > width) continue;
			glm::vec2 pt = glm::vec2(x, y);
			int pxIdx = y * width + x;

			glm::vec3 bary = calculateBarycentricCoordinate(tri, pt);
			if (!isBarycentricCoordInBounds(bary)) {
				//frags[pxIdx].color = glm::vec3(0);
				continue;
			}

			float zPersp = getZAtCoordinatePersp(bary, tri);
			glm::vec3 interNor = glm::normalize(getPerspectiveInterpolatedVector(bary, triNor, tri, zPersp));
			glm::vec3 interPos = getPerspectiveInterpolatedVector(bary, triPos, tri, zPersp);

			int depth = (int)( getZAtCoordinate(bary, tri) * INT_MAX);
			
			bool isSet;
			do {
				isSet = (atomicCAS(&mutex[pxIdx], 0, 1) == 0);
				if (isSet) {
					if (depthBuffer[pxIdx] > depth) {
						// replaced fragment with this triangle
						frags[pxIdx].color = interNor;
						frags[pxIdx].eyeNor = interNor;
						frags[pxIdx].eyePos = interPos;
						depthBuffer[pxIdx] = depth;

#if USE_TEXTURES
						if (pri.v[0].dev_diffuseTex != NULL) {
							glm::vec3 triUV[3] = {
								glm::vec3(pri.v[0].texcoord0, 0.f),
								glm::vec3(pri.v[1].texcoord0, 0.f),
								glm::vec3(pri.v[2].texcoord0, 0.f)
							};
							glm::vec2 interUV = glm::vec2(getPerspectiveInterpolatedVector(bary, triUV, tri, zPersp));

							frags[pxIdx].dev_diffuseTex = pri.v[0].dev_diffuseTex;
							frags[pxIdx].texcoord0 = interUV;
							frags[pxIdx].texHeight = pri.v[0].texHeight;
							frags[pxIdx].texWidth = pri.v[0].texWidth;
						}
						else {
							frags[pxIdx].dev_diffuseTex = NULL;
						}
						
#endif
					}	
					mutex[pxIdx] = 0;
				}
			} while (!isSet);
			
		}
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	checkCUDAError("init depth");
	initMutex << < blockCount2d, blockSize2d >> > (width, height, dev_fragMutex);
	checkCUDAError("init mutex");

	const int numThreads = 128;
	dim3 triBlockCount = (totalNumPrimitives + numThreads - 1) / numThreads;

	_rasterizeTriangle << < triBlockCount, numThreads >> > (totalNumPrimitives, dev_primitives, dev_fragmentBuffer, 
		dev_depth, width, height, dev_fragMutex);
	checkCUDAError("rasterize tris");

	

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");

#if BLOOM
	// make downsampled high pass
	dim3 blockDownsampleCount2d((width / 2 - 1) / blockSize2d.x + 1,
		(height / 2 - 1) / blockSize2d.y + 1);

	bloomHighPass << < blockDownsampleCount2d, blockSize2d >> > (width / 2, height / 2, dev_framebuffer, dev_bloom1);

	// apply gaussian
	bloomHorizontalGather << < blockDownsampleCount2d, blockSize2d >> >(width / 2, height / 2, dev_bloom1, dev_bloom2);
	bloomVerticalGather << < blockDownsampleCount2d, blockSize2d >> >(width / 2, height / 2, dev_bloom2, dev_bloom1);

#if BLOOM2PASS
	bloomHorizontalGather << < blockDownsampleCount2d, blockSize2d >> >(width / 2, height / 2, dev_bloom1, dev_bloom2);
	bloomVerticalGather << < blockDownsampleCount2d, blockSize2d >> >(width / 2, height / 2, dev_bloom2, dev_bloom1);
#endif

	// upsample and composite
	bloomComposite << < blockCount2d, blockSize2d >> > (width, height, dev_framebuffer, dev_bloom1);

#endif


	// HDR tonemap
	toneMap << <blockCount2d, blockSize2d >> >(width, height, dev_framebuffer, GAMMA, EXPOSURE);
	checkCUDAError("fragment shader");

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_fragMutex);
	dev_fragMutex = NULL;

#if BLOOM
	cudaFree(dev_bloom1);
	dev_bloom1 = NULL;
	cudaFree(dev_bloom2);
	dev_bloom2 = NULL;
#endif

    checkCUDAError("rasterize Free");
}
