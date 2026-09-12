// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/GPU/Metal/MetalShaderCompiler.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "ext/glslang/SPIRV/GlslangToSpv.h"

// The existing SPIRV-Cross targets abort on unsupported input. Metal uses a
// separate namespace and exception-enabled build so a shader failure is reported
// to the caller without changing the compiler used by the existing backends.
#define SPIRV_CROSS_NAMESPACE_OVERRIDE ppsspp_metal_spirv
#include "ext/SPIRV-Cross/spirv_msl.hpp"
#undef SPIRV_CROSS_NAMESPACE_OVERRIDE

namespace Metal {

bool CompileShader(std::string_view source, ShaderStage stage, const ShaderCompileOptions &options,
	CompiledShader *result, std::string *error) {
	if (!result || !error) {
		return false;
	}
	*result = {};
	error->clear();
	if (source.empty() || source.size() > (size_t)std::numeric_limits<int>::max()) {
		*error = "Metal shader source is empty or too large";
		return false;
	}

	EShLanguage language;
	spv::ExecutionModel model;
	switch (stage) {
	case ShaderStage::Vertex:
		language = EShLangVertex;
		model = spv::ExecutionModelVertex;
		break;
	case ShaderStage::Fragment:
		language = EShLangFragment;
		model = spv::ExecutionModelFragment;
		break;
	case ShaderStage::Compute:
		language = EShLangCompute;
		model = spv::ExecutionModelGLCompute;
		break;
	default:
		*error = "Unsupported Metal shader stage";
		return false;
	}

	// TProgram must be destroyed before the shaders linked into it.
	glslang::TShader shader(language);
	glslang::TProgram program;
	const char *text = source.data();
	int length = (int)source.size();
	shader.setStringsWithLengths(&text, &length, 1);
	shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 450);
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
	TBuiltInResource resources{};
	InitShaderResources(resources);
	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	if (!shader.parse(&resources, 450, ECoreProfile, false, false, messages)) {
		*error = std::string("Metal GLSL parse failed: ") + shader.getInfoLog() + shader.getInfoDebugLog();
		return false;
	}
	program.addShader(&shader);
	if (!program.link(messages)) {
		*error = std::string("Metal GLSL link failed: ") + program.getInfoLog() + program.getInfoDebugLog();
		return false;
	}
	std::vector<uint32_t> spirv;
	glslang::SpvOptions spvOptions{};
	glslang::GlslangToSpv(*program.getIntermediate(language), spirv, &spvOptions);

	try {
		ppsspp_metal_spirv::CompilerMSL compiler(std::move(spirv));
		ppsspp_metal_spirv::CompilerMSL::Options mslOptions;
		mslOptions.platform = options.ios ? ppsspp_metal_spirv::CompilerMSL::Options::iOS : ppsspp_metal_spirv::CompilerMSL::Options::macOS;
		mslOptions.set_msl_version(3, 0);
		compiler.set_msl_options(mslOptions);
		auto commonOptions = compiler.get_common_options();
		commonOptions.vertex.flip_vert_y = options.flipVertexY;
		commonOptions.vertex.fixup_clipspace = false;
		compiler.set_common_options(commonOptions);

		CompiledShader compiled;
		auto reflected = compiler.get_shader_resources();
		if (!reflected.push_constant_buffers.empty() || !reflected.subpass_inputs.empty() ||
			!reflected.separate_images.empty() || !reflected.separate_samplers.empty() ||
			!reflected.atomic_counters.empty()) {
			*error = "Metal shader uses a resource outside the shared shader binding layout";
			return false;
		}
		auto mapResources = [&](const auto &list, ResourceKind kind) {
			for (const auto &resource : list) {
				if (compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) != 0 ||
					!compiler.has_decoration(resource.id, spv::DecorationBinding) ||
					!compiler.get_type(resource.type_id).array.empty()) {
					throw std::runtime_error("Metal requires explicit set-0 bindings without descriptor arrays");
				}
				uint32_t binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
				bool buffer = kind == ResourceKind::UniformBuffer || kind == ResourceKind::StorageBuffer;
				uint32_t index = binding;
				uint32_t byteSize = 0;
				if (buffer) {
					if (index >= VERTEX_BUFFER_SLOT) {
						throw std::runtime_error("Metal shader buffer binding conflicts with reserved buffer slots");
					}
					byteSize = (uint32_t)compiler.get_declared_struct_size(compiler.get_type(resource.base_type_id));
				} else {
					if (binding < options.textureBindingBase || binding - options.textureBindingBase >= 16) {
						throw std::runtime_error("Metal shader texture binding is outside the supported layout");
					}
					index -= options.textureBindingBase;
				}
				ppsspp_metal_spirv::MSLResourceBinding mapping;
				mapping.stage = model;
				mapping.desc_set = 0;
				mapping.binding = binding;
				mapping.msl_buffer = index;
				mapping.msl_texture = index;
				mapping.msl_sampler = index;
				compiler.add_msl_resource_binding(mapping);
				compiled.resources.push_back({ kind, binding, index, byteSize });
			}
		};
		mapResources(reflected.uniform_buffers, ResourceKind::UniformBuffer);
		mapResources(reflected.storage_buffers, ResourceKind::StorageBuffer);
		mapResources(reflected.sampled_images, ResourceKind::SampledTexture);
		mapResources(reflected.storage_images, ResourceKind::StorageTexture);

		compiled.source = compiler.compile();
		for (const auto &name : options.noInlineFunctions) {
			const std::string attribute = "static inline __attribute__((always_inline))\n";
			bool found = false;
			for (size_t pos = compiled.source.find(attribute); pos != std::string::npos; pos = compiled.source.find(attribute, pos + 1)) {
				const size_t signature = pos + attribute.size();
				const size_t end = compiled.source.find('\n', signature);
				if (compiled.source.substr(signature, end - signature).find(" " + name + "(") != std::string::npos) {
					compiled.source.replace(pos, attribute.size(), "static __attribute__((noinline))\n");
					found = true;
					break;
				}
			}
			if (!found) {
				*error = "Metal out-of-line helper not found: " + name;
				return false;
			}
		}
		compiled.entryPoint = compiler.get_cleansed_entry_point_name("main", model);
		if (compiler.needs_swizzle_buffer() || compiler.needs_buffer_size_buffer() || compiler.needs_view_mask_buffer() ||
			compiler.needs_output_buffer() || compiler.needs_patch_output_buffer()) {
			*error = "Metal shader requires unsupported auxiliary buffers";
			return false;
		}
		if (stage == ShaderStage::Compute) {
			for (int i = 0; i < 3; ++i) {
				compiled.workgroupSize[i] = compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, i);
			}
		}
		*result = std::move(compiled);
		return true;
	} catch (const std::exception &exception) {
		*error = std::string("Metal shader translation failed: ") + exception.what();
		return false;
	}
}

}  // namespace Metal
