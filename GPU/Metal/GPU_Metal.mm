// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/GPU_Metal.h"
#include "GPU/Metal/DrawEngineMetal.h"
#include "GPU/Metal/FramebufferManagerMetal.h"
#include "GPU/GPUCommonHW.h"
#include "Common/Data/Text/StringWriter.h"

class GPU_Metal final : public GPUCommonHW {
public:
	GPU_Metal(GraphicsContext *context, Draw::DrawContext *draw) : GPUCommonHW(context, draw), drawEngine_(draw) {
		metalShaders_ = new ShaderManagerMetal(draw);
		auto framebuffer = new FramebufferManagerMetal(draw);
		auto textures = new TextureCacheMetal(draw, framebuffer->GetDraw2D());
		shaderManager_ = metalShaders_;
		framebufferManager_ = framebuffer;
		textureCache_ = textures;
		drawEngineCommon_ = &drawEngine_;
		drawEngine_.SetGPUCommon(this);
		drawEngine_.SetShaderManager(metalShaders_);
		drawEngine_.SetTextureCache(textures);
		drawEngine_.SetFramebufferManager(framebuffer);
		drawEngine_.Init();
		framebuffer->SetShaderManager(metalShaders_);
		framebuffer->SetTextureCache(textures);
		framebuffer->SetDrawEngine(&drawEngine_);
		framebuffer->Init(msaaLevel_);
		textures->SetFramebufferManager(framebuffer);
		textures->SetShaderManager(metalShaders_);
		UpdateCmdInfo();
		gstate_c.SetUseFlags(CheckGPUFeatures());
		BuildReportingInfo();
		textures->NotifyConfigChanged();
	}

	~GPU_Metal() override {
		// GPUCommonHW owns the managers and deletes them after member destruction.
		framebufferManager_->SetDrawEngine(nullptr);
	}

	u32 CheckGPUFeatures() const override {
		u32 features = GPUCommonHW::CheckGPUFeatures();
		if ((draw_->GetDataFormatSupport(Draw::DataFormat::R5G6B5_UNORM_PACK16) & Draw::FMT_TEXTURE) &&
			(draw_->GetDataFormatSupport(Draw::DataFormat::R5G5B5A1_UNORM_PACK16) & Draw::FMT_TEXTURE) &&
			(draw_->GetDataFormatSupport(Draw::DataFormat::R4G4B4A4_UNORM_PACK16) & Draw::FMT_TEXTURE)) {
			features |= GPU_USE_16BIT_FORMATS;
		}
		return CheckGPUFeaturesLate(features);
	}

	void GetStats(StringWriter &writer) override {
		FormatGPUStatsCommon(writer);
		writer.F("Metal vertex/fragment shaders: %d / %d\n", metalShaders_->GetNumVertexShaders(), metalShaders_->GetNumFragmentShaders());
		writer.F("Metal MSAA samples: %d\n", 1 << msaaLevel_);
	}

protected:
	void FinishDeferred() override { drawEngine_.FlushPartialDecode(); }

private:
	void BeginHostFrame(const DisplayLayoutConfig &config) override {
		GPUCommonHW::BeginHostFrame(config);
		textureCache_->StartFrame();
		drawEngine_.BeginFrame();
		framebufferManager_->BeginFrame(config);
		if (gstate_c.useFlagsChanged) {
			shaderManager_->ClearShaders();
			framebufferManager_->ClearAllDepthBuffers();
			gstate_c.useFlagsChanged = false;
		}
	}

	DrawEngineMetal drawEngine_;
	ShaderManagerMetal *metalShaders_ = nullptr;
};

GPUCommon *CreateMetalGPU(GraphicsContext *context, Draw::DrawContext *draw) {
	return new GPU_Metal(context, draw);
}
