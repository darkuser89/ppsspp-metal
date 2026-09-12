// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/Metal/ShaderManagerMetal.h"
#include "GPU/Common/VertexDecoderCommon.h"

struct MetalGEBlendState {
	bool enabled = false;
	MTLBlendFactor srcColor = MTLBlendFactorOne;
	MTLBlendFactor dstColor = MTLBlendFactorZero;
	MTLBlendOperation colorOp = MTLBlendOperationAdd;
	MTLBlendFactor srcAlpha = MTLBlendFactorOne;
	MTLBlendFactor dstAlpha = MTLBlendFactorZero;
	MTLBlendOperation alphaOp = MTLBlendOperationAdd;
	MTLColorWriteMask writeMask = MTLColorWriteMaskAll;
};

struct MetalGEDepthStencilState {
	bool depthWrite = false;
	MTLCompareFunction depthCompare = MTLCompareFunctionAlways;
	bool stencilEnabled = false;
	MTLCompareFunction stencilCompare = MTLCompareFunctionAlways;
	MTLStencilOperation stencilFail = MTLStencilOperationKeep;
	MTLStencilOperation depthFail = MTLStencilOperationKeep;
	MTLStencilOperation pass = MTLStencilOperationKeep;
	uint8_t readMask = 255;
	uint8_t writeMask = 255;
};

struct MetalGEPipeline {
	id<MTLRenderPipelineState> state = nil;
	uint32_t stride = 0;
	uint32_t textureMask = 0;
};

// Native GE pipelines use the shared decoder layout for hardware transform and
// TransformedVertex for CPU transform. Raster, viewport, stencil reference and
// blend constants are dynamic encoder state, not render pipeline identity.
class PipelineManagerMetal {
public:
	explicit PipelineManagerMetal(Metal::RenderManager *manager) : manager_(manager) {}
	void Clear();
	void DeviceLost();
	void DeviceRestore(Metal::RenderManager *manager);

	// The result is owned by this cache until Clear/DeviceLost or a shader cache
	// generation change. Already encoded commands retain their native PSOs.
	const MetalGEPipeline *GetOrCreate(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
		const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
		MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount = 1);
	id<MTLDepthStencilState> GetDepthStencil(const MetalGEDepthStencilState &state, std::string *error);
	int GetNumPipelines() const { return (int)pipelines_.size(); }
	int GetNumDepthStencilStates() const { return (int)depthStates_.size(); }

private:
	// Explicit integer fields avoid struct padding and bitfield layout in keys.
	using PipelineKey = std::array<uint64_t, 21>;
	using DepthKey = std::array<uint64_t, 9>;
	Metal::RenderManager *manager_ = nullptr;
	const ShaderManagerMetal *shaderOwner_ = nullptr;
	uint64_t shaderGeneration_ = 0;
	std::map<PipelineKey, MetalGEPipeline> pipelines_;
	std::map<DepthKey, id<MTLDepthStencilState>> depthStates_;
};
