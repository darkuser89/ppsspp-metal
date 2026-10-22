// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/metal4/ShaderManagerMetal.h"
// Foundation's ABS macro conflicts with the ARM64 emitter instruction.
#pragma push_macro("ABS")
#undef ABS
#include "GPU/Common/VertexDecoderCommon.h"
#pragma pop_macro("ABS")

#include <condition_variable>
#include <array>
#include <memory>
#include <mutex>
#include <vector>

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
	MTL4RenderPipelineDescriptor *descriptor = nil;
	uint32_t stride = 0;
	uint32_t textureMask = 0;
};

// Native compiler elapsed time; asynchronous intervals may overlap.
struct MetalPipelineMetrics {
	uint64_t syncCompiles = 0;
	uint64_t asyncCompiles = 0;
	uint64_t waits = 0;
	uint64_t archiveSaves = 0;
	uint64_t archivePipelines = 0;
	uint64_t cacheSaves = 0;
	uint64_t archiveDeferrals = 0;
	uint64_t archiveFailures = 0;
	double syncSeconds = 0.0;
	double asyncSeconds = 0.0;
	double waitSeconds = 0.0;
	double archiveSeconds = 0.0;
	double cacheSaveSeconds = 0.0;
};

// Native GE pipelines use the shared decoder layout for hardware transform and
// TransformedVertex for CPU transform. Raster, viewport, stencil reference and
// blend constants are dynamic encoder state, not render pipeline identity.
class PipelineManagerMetal {
public:
	explicit PipelineManagerMetal(Metal::RenderManager *manager) : manager_(manager) {}
	// Called on the render owner thread; completion callbacks only update each entry.
	bool PipelinesPending() const;
	MetalPipelineMetrics Metrics() const;
	void Clear();
	void DeviceLost();
	void DeviceRestore(Metal::RenderManager *manager);
	bool LoadCache(const Path &filename, ShaderManagerMetal &shaders);
	void SaveCache(const Path &filename) const;

	// The result is owned by this cache until Clear/DeviceLost or a shader cache
	// generation change. RenderContext retains encoded PSOs until GPU completion.
	const MetalGEPipeline *GetOrCreate(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
		const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
		MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount = 1);
	const MetalGEPipeline *Request(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
		const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
		MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount = 1);
	id<MTLDepthStencilState> GetDepthStencil(const MetalGEDepthStencilState &state, std::string *error);
	int GetNumPipelines() const { return (int)(pipelines_.size() + pending_.size()); }
	int GetNumDepthStencilStates() const { return (int)depthStates_.size(); }

private:
	// Explicit integer fields avoid struct padding and bitfield layout in keys.
	using PipelineKey = std::array<uint64_t, 21>;
	using DepthKey = std::array<uint64_t, 9>;
	struct PendingPipeline {
		std::mutex mutex;
		std::condition_variable ready;
		MetalGEPipeline pipeline;
		std::string error;
		bool complete = false;
	};
	const MetalGEPipeline *GetOrCreateInternal(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
		const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
		MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount, bool async);
	struct MetricsState {
		std::mutex mutex;
		MetalPipelineMetrics value;
		uint32_t activeCompiles = 0;
	};
	// Native callbacks can outlive Clear/DeviceLost and the owner itself.
	std::shared_ptr<MetricsState> metrics_ = std::make_shared<MetricsState>();
	Metal::RenderManager *manager_ = nullptr;
	const ShaderManagerMetal *shaderOwner_ = nullptr;
	uint64_t shaderGeneration_ = 0;
	PipelineKey lastKey_{};
	const MetalGEPipeline *lastPipeline_ = nullptr;
	std::map<PipelineKey, MetalGEPipeline> pipelines_;
	std::map<PipelineKey, std::shared_ptr<PendingPipeline>> pending_;
	std::map<DepthKey, id<MTLDepthStencilState>> depthStates_;
#if PPSSPP_PLATFORM(MAC)
	mutable NSArray<id<MTL4Archive>> *binaryArchives_ = nil;
	mutable uint32_t binaryArchiveBank_ = 0;
	mutable Path binaryArchiveCachePath_;
	mutable std::vector<PipelineKey> binaryArchiveKeys_;
	mutable std::array<uint64_t, 8> binaryArchiveHashes_{};
#endif
};
