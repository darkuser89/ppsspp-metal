// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/Common/DrawEngineCommon.h"
#include "GPU/metal4/StateMappingMetal.h"
#include "GPU/metal4/TextureCacheMetal.h"

class FramebufferManagerMetal;

struct MetalTransformMetrics {
	uint64_t flatDraws = 0;
	uint64_t flatHardwareEligible = 0;
	uint64_t flatHardwareDraws = 0;
	uint64_t flatLightingFallbacks = 0;
	uint64_t flatVertices = 0;
	double flatPreparationSeconds = 0.0;
};

class DrawEngineMetal : public DrawEngineCommon {
public:
	explicit DrawEngineMetal(Draw::DrawContext *draw);
	~DrawEngineMetal() override;
	void DeviceLost() override;
	void DeviceRestore(Draw::DrawContext *draw) override;
	void BeginFrame() override;
	void Flush() override;
	void NotifyConfigChanged() override;
	void SetShaderManager(ShaderManagerMetal *manager) { shaderManager_ = manager; }
	void SetTextureCache(TextureCacheMetal *cache) { textureCache_ = cache; }
	void SetFramebufferManager(FramebufferManagerMetal *manager) { framebufferManager_ = manager; }
	void LoadPipelineCache(const Path &filename) {
		if (shaderManager_) {
			pipelines_.LoadCache(filename, *shaderManager_);
		}
	}
	void SavePipelineCache(const Path &filename) const { pipelines_.SaveCache(filename); }
	bool PipelinesPending() const { return pipelines_.PipelinesPending(); }
	int GetNumPipelines() const { return pipelines_.GetNumPipelines(); }
	MetalPipelineMetrics PipelineMetrics() const { return pipelines_.Metrics(); }
	const std::string &LastError() const { return lastError_; }
	const MetalTransformMetrics &TransformMetrics() const { return transformMetrics_; }

private:
	struct EncoderState {
		bool valid = false;
		uint64_t serial = 0;
		id<MTLRenderPipelineState> pipeline = nil;
		id<MTLDepthStencilState> depth = nil;
		NSUInteger stencil = 0;
		MTLCullMode cull = MTLCullModeNone;
		MTLDepthClipMode depthClip = MTLDepthClipModeClip;
		MTLViewport viewport{};
		MTLScissorRect scissor{};
		std::array<float, 4> blendColor{};
	};
	bool FlushDraw(std::string *error);
	bool ApplyDrawState(GEPrimitiveType prim, MetalDrawState *state, std::string *error);
	void Invalidate(InvalidationCallbackFlags flags);
	Draw::DrawContext *draw_ = nullptr;
	Metal::RenderManager *manager_ = nullptr;
	PipelineManagerMetal pipelines_{nullptr};
	ShaderManagerMetal *shaderManager_ = nullptr;
	TextureCacheMetal *textureCache_ = nullptr;
	FramebufferManagerMetal *framebufferManager_ = nullptr;
	std::string lastError_;
#if PPSSPP_PLATFORM(MAC)
	std::vector<uint16_t> softwareIndices_;
	std::vector<std::array<int, 4>> samplingRegions_;
#endif
	EncoderState encoderState_;
	MetalTransformMetrics transformMetrics_;
};
