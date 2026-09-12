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
	const std::string &LastError() const { return lastError_; }

private:
	bool FlushDraw(std::string *error);
	bool ApplyDrawState(GEPrimitiveType prim, MetalDrawState *state, ViewportAndScissor *viewport, std::string *error);
	void Invalidate(InvalidationCallbackFlags flags);
	Draw::DrawContext *draw_ = nullptr;
	Metal::RenderManager *manager_ = nullptr;
	PipelineManagerMetal pipelines_{nullptr};
	SamplerCacheMetal samplers_;
	ShaderManagerMetal *shaderManager_ = nullptr;
	TextureCacheMetal *textureCache_ = nullptr;
	FramebufferManagerMetal *framebufferManager_ = nullptr;
	std::string lastError_;
};
