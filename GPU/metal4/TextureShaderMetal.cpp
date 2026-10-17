// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#include "GPU/metal4/TextureShaderMetal.h"
const char *metalUploadShader = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable

// 8x8 is the most common compute shader workgroup size, and works great on all major
// hardware vendors. TODO: However, we should probably change to 16x16, as Qualcomm now has
// support for bigger groups...
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

uniform layout(set = 0, binding = 0, rgba8) writeonly image2D img;

layout(std430, set = 0, binding = 1) buffer Buf {
	uint data[];
} buf;

layout(push_constant) uniform Params {
	int width;
	int height;
} params;

// The cbuffer, if present, is self-declared for layout flexibility
#define CBUFFER_SET 0
#define CBUFFER_BINDING 4

uint readColoru(uvec2 p) {
	return buf.data[p.y * params.width + p.x];
}

vec4 readColorf(uvec2 p) {
	// Unpack the color (we could look it up in a CLUT here if we wanted...)
	// The imageStore repack is free.
	return unpackUnorm4x8(readColoru(p));
}

void writeColorf(ivec2 p, vec4 c) {
	imageStore(img, p, c);
}

%s

// Note that main runs once per INPUT pixel, unlike the old model.
void main() {
	uvec2 xy = gl_GlobalInvocationID.xy;
	// Kill off any out-of-image threads to avoid stray writes.
	// Should only happen on the tiniest mipmaps as PSP textures are power-of-2,
	// and we use a 8x8 workgroup size. Probably not really necessary.
	if (xy.x >= params.width || xy.y >= params.height)
		return;
	// applyScaling will write the upscaled pixels, using writeColorf above.
	// It's expected to write a square of scale*scale pixels, at the location xy*scale.
	applyScaling(xy);
}

)";
