// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/Common/DrawEngineCommon.h"
#include "GPU/Metal/StateMappingMetal.h"
#include "GPU/Metal/TextureCacheMetal.h"

class FramebufferManagerMetal;

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
	int GetNumPipelines() const { return pipelines_.GetNumPipelines(); }
	const std::string &LastError() const { return lastError_; }

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
		uint32_t blendColor = 0;
	};
	bool FlushDraw(std::string *error);
	bool ApplyDrawState(GEPrimitiveType prim, MetalDrawState *state, std::string *error);
	void Invalidate(InvalidationCallbackFlags flags);
	Draw::DrawContext *draw_ = nullptr;
	Metal::RenderManager *manager_ = nullptr;
	PipelineManagerMetal pipelines_{nullptr};
	SamplerCacheMetal samplers_;
	ShaderManagerMetal *shaderManager_ = nullptr;
	TextureCacheMetal *textureCache_ = nullptr;
	FramebufferManagerMetal *framebufferManager_ = nullptr;
	std::string lastError_;
	EncoderState encoderState_;
};
