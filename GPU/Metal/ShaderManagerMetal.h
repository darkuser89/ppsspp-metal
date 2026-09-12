// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <memory>

#include "Common/GPU/Metal/MetalRenderManager.h"
#include "GPU/Common/ShaderCommon.h"
#include "GPU/Common/ShaderId.h"
#include "GPU/Common/ShaderUniforms.h"
#include "GPU/Common/VertexShaderGenerator.h"
#include "GPU/Common/FragmentShaderGenerator.h"

struct MetalGEShader {
	id<MTLFunction> function = nil;
	Metal::CompiledShader compiled;
	std::string source;
	std::string error;
	uint64_t uniformMask = 0;
};

struct MetalGEVertexShader : MetalGEShader {
	VShaderID id;
	uint32_t attributeMask = 0;
	VertexShaderFlags flags{};
	bool UseHWTransform() const { return id.Bit(VS_BIT_USE_HW_TRANSFORM); }
};

struct MetalGEFragmentShader : MetalGEShader {
	FShaderID id;
	FragmentShaderFlags flags{};
};

struct MetalGEUniformBuffers {
	id<MTLBuffer> base = nil;
	id<MTLBuffer> lights = nil;
};

// Owned by the emulation/render thread, like the shared generators and gstate.
// A returned shader remains valid until ClearShaders/DeviceLost or an environment
// change. Pipeline caches must include CacheGeneration(), not just the shader IDs.
class ShaderManagerMetal final : public ShaderManagerCommon {
public:
	explicit ShaderManagerMetal(Draw::DrawContext *draw);
	~ShaderManagerMetal() override = default;
	void ClearShaders() override;
	void DeviceLost() override;
	void DeviceRestore(Draw::DrawContext *draw) override;

	bool GetShaders(u32 vertexType, const ComputedPipelineState &pipelineState, bool useHWTransform,
		ClipInfoFlags clipInfoFlags, const MetalGEVertexShader **vertex, const MetalGEFragmentShader **fragment, std::string *error);
	const MetalGEVertexShader *GetVertexShaderFromID(VShaderID id, std::string *error);
	const MetalGEFragmentShader *GetFragmentShaderFromID(FShaderID id, std::string *error);
	uint64_t CacheGeneration() const { return generation_; }
	int GetNumVertexShaders() const { return (int)vertexCache_.size(); }
	int GetNumFragmentShaders() const { return (int)fragmentCache_.size(); }

	// Uploads immutable snapshots. Failure retains the previous snapshots and all
	// dirty flags. Encoded draws retain their old buffers across later updates.
	bool UpdateUniforms(bool useBufferedRendering, bool pixelMapped, std::string *error);
	void SetSamplerLodBias(float bias);
	bool BindUniforms(id<MTLRenderCommandEncoder> encoder, std::string *error) const;
	const MetalGEUniformBuffers &Uniforms() const { return buffers_; }

	std::vector<std::string> DebugGetShaderIDs(DebugShaderType type) override;
	std::string DebugGetShaderString(std::string id, DebugShaderType type, DebugShaderStringType stringType) override;

private:
	bool CheckEnvironment(std::string *error);
	bool Compile(MetalGEShader *shader, ShaderStage stage, const char *tag);
	Metal::RenderManager *manager_ = nullptr;
	ShaderLanguageDesc language_{GLSL_VULKAN};
	std::map<VShaderID, std::unique_ptr<MetalGEVertexShader>> vertexCache_;
	std::map<FShaderID, std::unique_ptr<MetalGEFragmentShader>> fragmentCache_;
	std::array<char, 32768> code_{};
	uint64_t generation_ = 0;
	u32 useFlags_ = 0;
	bool vendorChecks_ = false;
	UB_VS_FS_Base base_{};
	UB_VS_Lights lights_{};
	MetalGEUniformBuffers buffers_;
	bool uniformOptionsValid_ = false;
	bool buffered_ = false;
	bool pixelMapped_ = false;
	float samplerLodBias_ = 0.0f;
	bool samplerLodBiasDirty_ = true;
};
