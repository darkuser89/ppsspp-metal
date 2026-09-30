// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/GPU_Metal.h"
#include "GPU/Metal/DrawEngineMetal.h"
#include "GPU/Metal/FramebufferManagerMetal.h"
#include "GPU/GPUCommonHW.h"
#include "Common/GPU/Metal/MetalResources.h"
#include "Common/Data/Text/StringWriter.h"
#include "Common/File/FileUtil.h"
#include "Core/Config.h"
#include "Core/ELF/ParamSFO.h"
#include "Core/Reporting.h"
#include "Core/System.h"

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
		const std::string discID = g_paramSFO.GetDiscID();
		if (g_Config.bShaderCache && !discID.empty()) {
			File::CreateFullPath(GetSysDirectory(DIRECTORY_APP_CACHE));
			shaderCachePath_ = GetSysDirectory(DIRECTORY_APP_CACHE) / (discID + ".metalshadercache");
			pipelineCachePath_ = GetSysDirectory(DIRECTORY_APP_CACHE) / (discID + ".metalpipelinecache");
			metalShaders_->LoadCache(shaderCachePath_);
			drawEngine_.LoadPipelineCache(pipelineCachePath_);
		}
	}

	~GPU_Metal() override {
		if (draw_) {
			drawEngine_.SavePipelineCache(pipelineCachePath_);
			metalShaders_->SaveCache(shaderCachePath_);
		}
		// GPUCommonHW owns the managers and deletes them after member destruction.
		framebufferManager_->SetDrawEngine(nullptr);
	}

	u32 CheckGPUFeatures() const override {
		u32 features = GPUCommonHW::CheckGPUFeatures();
		// On macOS Apple GPUs, multisample fetch reads the current color for each
		// sample. Keep the copy path for layered targets and for iOS MSAA.
#if PPSSPP_PLATFORM(MAC)
		const bool multisampleFetchSupported = true;
#else
		const bool multisampleFetchSupported = false;
#endif
		if ((msaaLevel_ != 0 && !multisampleFetchSupported) || g_Config.bStereoRendering) {
			features &= ~GPU_USE_FRAMEBUFFER_FETCH;
			// Unbuffered draws can still read the single-sample backbuffer through
			// SnapshotBackbufferColor() when shader blending requires it.
		}
		if ((draw_->GetDataFormatSupport(Draw::DataFormat::R5G6B5_UNORM_PACK16) & Draw::FMT_TEXTURE) &&
			(draw_->GetDataFormatSupport(Draw::DataFormat::R5G5B5A1_UNORM_PACK16) & Draw::FMT_TEXTURE) &&
			(draw_->GetDataFormatSupport(Draw::DataFormat::R4G4B4A4_UNORM_PACK16) & Draw::FMT_TEXTURE)) {
			features |= GPU_USE_16BIT_FORMATS;
		}
		features |= GPU_USE_FRAMEBUFFER_ARRAYS;
		id<MTLDevice> device = (__bridge id<MTLDevice>)(void *)draw_->GetNativeObject(Draw::NativeObject::DEVICE);
		bool layeredMSAA = msaaLevel_ == 0;
		if (@available(macOS 13.0, iOS 16.0, *)) {
			layeredMSAA |= [device supportsFamily:MTLGPUFamilyApple7] || [device supportsFamily:MTLGPUFamilyMac2];
		}
		if (g_Config.bStereoRendering && draw_->GetDeviceCaps().multiViewSupported && layeredMSAA) {
			features |= GPU_USE_SINGLE_PASS_STEREO;
			features |= GPU_USE_SIMPLE_STEREO_PERSPECTIVE;
		}
		return CheckGPUFeaturesLate(features);
	}

	void GetStats(StringWriter &writer) override {
		FormatGPUStatsCommon(writer);
		writer.F("Metal vertex/fragment shaders: %d / %d\n", metalShaders_->GetNumVertexShaders(), metalShaders_->GetNumFragmentShaders());
		writer.F("Metal GE pipelines: %d\n", drawEngine_.GetNumPipelines());
		writer.F("Metal MSAA samples: %d\n", 1 << msaaLevel_);
		if (draw_) {
			writer.F("Metal render passes begun this frame: %u\n", draw_->GetCurrentBackendState().passes);
			auto *context = (Metal::RenderContext *)draw_->GetNativeObject(Draw::NativeObject::CONTEXT);
			if (context) {
				const double gpuTimeMs = context->LastSubmissionGPUTimeMs();
				if (gpuTimeMs >= 0.0) {
					writer.F("Last Metal submission GPU time: %.3f ms\n", gpuTimeMs);
				}
			}
		}
	}

	void BuildReportingInfo() override {
		id<MTLDevice> device = (__bridge id<MTLDevice>)(void *)draw_->GetNativeObject(Draw::NativeObject::DEVICE);
		if (!device) {
			GPUCommonHW::BuildReportingInfo();
			return;
		}
		reportingPrimaryInfo_ = device.name.UTF8String;
		reportingFullInfo_ = "Metal 3, MSL 3.0 (" + reportingPrimaryInfo_;
		if ([device supportsFamily:MTLGPUFamilyApple7]) {
			reportingFullInfo_ += ", Apple7+";
		} else if ([device supportsFamily:MTLGPUFamilyApple5]) {
			reportingFullInfo_ += ", Apple5+";
		} else if ([device supportsFamily:MTLGPUFamilyMac2]) {
			reportingFullInfo_ += ", Mac2";
		}
		if (device.supportsBCTextureCompression) {
			reportingFullInfo_ += ", BC textures";
		}
		if (draw_->GetDeviceCaps().multiViewSupported) {
			reportingFullInfo_ += ", multiview";
		}
		if (Metal::SupportsDepthStencilResolve(device)) {
			reportingFullInfo_ += ", depth/stencil resolve";
		}
		reportingFullInfo_ += ")";
		Reporting::UpdateConfig();
	}

protected:
	void FinishDeferred() override { drawEngine_.Flush(); }
	void DeviceLost() override {
		drawEngine_.SavePipelineCache(pipelineCachePath_);
		metalShaders_->SaveCache(shaderCachePath_);
		GPUCommonHW::DeviceLost();
	}
	void DeviceRestore(Draw::DrawContext *draw) override {
		GPUCommonHW::DeviceRestore(draw);
		metalShaders_->LoadCache(shaderCachePath_);
		drawEngine_.LoadPipelineCache(pipelineCachePath_);
	}

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
		if (shaderCachePath_.Valid() && gpuStats.totals.numFlips && !(gpuStats.totals.numFlips & 32767)) {
			drawEngine_.SavePipelineCache(pipelineCachePath_);
			metalShaders_->SaveCache(shaderCachePath_);
		}
	}

	DrawEngineMetal drawEngine_;
	ShaderManagerMetal *metalShaders_ = nullptr;
	Path shaderCachePath_;
	Path pipelineCachePath_;
};

GPUCommon *CreateMetalGPU(GraphicsContext *context, Draw::DrawContext *draw) {
	return new GPU_Metal(context, draw);
}
