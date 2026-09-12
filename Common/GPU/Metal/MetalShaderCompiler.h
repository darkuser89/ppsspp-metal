// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "Common/GPU/Shader.h"

namespace Metal {

// Uniform/storage buffers retain their explicit GLSL binding. Reserve one slot
// for vertex data, outside the range used by the shared PSP shader generators.
constexpr uint32_t VERTEX_BUFFER_SLOT = 16;

enum class ResourceKind {
	UniformBuffer,
	StorageBuffer,
	SampledTexture,
	StorageTexture,
};

struct ShaderResource {
	ResourceKind kind;
	uint32_t binding;
	uint32_t index;
	uint32_t byteSize;
};

struct ShaderCompileOptions {
	// Thin3D uses binding 0 for uniforms and starts textures at 1. PSP shaders
	// start textures at 0 and use bindings 3+ for uniforms.
	uint32_t textureBindingBase = 1;
	// GLSL_VULKAN and Metal both use 0..1 depth, but opposite viewport Y signs.
	bool flipVertexY = true;
	bool ios = false;
	// Keep large compute helpers out of line to avoid pathological pipeline
	// compilation after SPIRV-Cross's default forced inlining.
	std::vector<std::string> noInlineFunctions;
};

struct CompiledShader {
	std::string source;
	std::string entryPoint;
	std::vector<ShaderResource> resources;
	uint32_t workgroupSize[3] = { 1, 1, 1 };
};

// Consumes the GLSL 450 produced by the shared shader generators, not arbitrary
// SPIR-V. ShaderTranslationInit must precede calls, and ShaderTranslationShutdown
// must follow completion of all shader compilation jobs.
// Does not create a Vulkan instance or use Vulkan types.
bool CompileShader(std::string_view source, ShaderStage stage, const ShaderCompileOptions &options,
	CompiledShader *result, std::string *error);

}  // namespace Metal
