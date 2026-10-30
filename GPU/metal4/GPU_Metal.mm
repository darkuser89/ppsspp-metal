// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/metal4/GPU_Metal.h"
#include "GPU/metal4/DrawEngineMetal.h"
#include "GPU/metal4/FramebufferManagerMetal.h"
#include "GPU/metal4/GPUCommonHWMetal.h"
#include "Common/GPU/metal4/MetalResources.h"
#include "Common/Data/Text/StringWriter.h"
#include "Common/File/FileUtil.h"
#include "Core/Config.h"
#include "Core/ELF/ParamSFO.h"
#include "Core/Reporting.h"
#include "Core/System.h"

class GPU_Metal final : public GPUCommonHWMetal {
public:
	GPU_Metal(GraphicsContext *context, Draw::DrawContext *draw) : GPUCommonHWMetal(context, draw), drawEngine_(draw) {
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
		if (g_Config.bGpuLogProfiler) {
			char stats[8192];
			StringWriter writer(stats);
			GetStats(writer);
			INFO_LOG(Log::G3D, "Metal final profile counters:\n%s", stats);
		}
		// GPUCommonHWMetal owns the managers and deletes them after member destruction.
		framebufferManager_->SetDrawEngine(nullptr);
	}

	bool IsLoadingShaderCache() const override { return drawEngine_.PipelinesPending(); }

	u32 CheckGPUFeatures() const override {
		u32 features = GPUCommonHWMetal::CheckGPUFeatures();
#if PPSSPP_PLATFORM(MAC)
		// GE depth is 16-bit (gpu/exact/planes). Quantize before D32F storage
		// so coplanar passes do not compare the rasterizer's extra precision.
		features |= GPU_ROUND_FRAGMENT_DEPTH_TO_16BIT;
#endif
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
		// On macOS Apple GPUs, multisample fetch reads the current color for each
		// sample. Layered fetch requires buffered single-pass stereo.
#if PPSSPP_PLATFORM(MAC)
		const bool multisampleFetchSupported = true;
		const bool stereoFetchSupported = (features & GPU_USE_SINGLE_PASS_STEREO) != 0 && !g_Config.bSkipBufferEffects;
#else
		const bool multisampleFetchSupported = false;
		const bool stereoFetchSupported = false;
#endif
		if ((msaaLevel_ != 0 && !multisampleFetchSupported) || (g_Config.bStereoRendering && !stereoFetchSupported)) {
			features &= ~GPU_USE_FRAMEBUFFER_FETCH;
			// Unbuffered draws can still read the single-sample backbuffer through
			// SnapshotBackbufferColor() when shader blending requires it.
		}
		if ((draw_->GetDataFormatSupport(Draw::DataFormat::R5G6B5_UNORM_PACK16) & Draw::FMT_TEXTURE) &&
			(draw_->GetDataFormatSupport(Draw::DataFormat::R5G5B5A1_UNORM_PACK16) & Draw::FMT_TEXTURE) &&
			(draw_->GetDataFormatSupport(Draw::DataFormat::R4G4B4A4_UNORM_PACK16) & Draw::FMT_TEXTURE)) {
			features |= GPU_USE_16BIT_FORMATS;
		}
		features = CheckGPUFeaturesLate(features);
#if PPSSPP_PLATFORM(MAC)
		// Preserve explicit vertex-rounding overrides without rounding twice.
		if (features & GPU_ROUND_DEPTH_TO_16BIT) {
			features &= ~GPU_ROUND_FRAGMENT_DEPTH_TO_16BIT;
		}
#endif
		return features;
	}

	void GetStats(StringWriter &writer) override {
		FormatGPUStatsCommon(writer);
		writer.F("Metal vertex/fragment shaders: %d / %d\n", metalShaders_->GetNumVertexShaders(), metalShaders_->GetNumFragmentShaders());
		writer.F("Metal GE pipelines: %d\n", drawEngine_.GetNumPipelines());
		const auto pipelines = drawEngine_.PipelineMetrics();
		writer.F("Metal PSO sync/async elapsed totals: %.3f/%.3f ms (%llu/%llu compiles)\n",
			pipelines.syncSeconds * 1000.0, pipelines.asyncSeconds * 1000.0,
			(unsigned long long)pipelines.syncCompiles, (unsigned long long)pipelines.asyncCompiles);
		writer.F("Metal PSO owner waits: %.3f ms (%llu); archive captures: %.3f ms (%llu)\n",
			pipelines.waitSeconds * 1000.0, (unsigned long long)pipelines.waits,
			pipelines.archiveSeconds * 1000.0, (unsigned long long)pipelines.archiveSaves);
		writer.F("Metal pipeline cache saves: %.3f ms (%llu calls, %llu deferred); %llu binaries captured, %llu archive failures\n",
			pipelines.cacheSaveSeconds * 1000.0, (unsigned long long)pipelines.cacheSaves,
			(unsigned long long)pipelines.archiveDeferrals, (unsigned long long)pipelines.archivePipelines, (unsigned long long)pipelines.archiveFailures);
		const auto &compile = metalShaders_->Metrics();
		writer.F("Metal shader generation/frontend CPU totals: %.3f/%.3f ms (%llu/%llu calls)\n",
			compile.generationSeconds * 1000.0, compile.frontendSeconds * 1000.0,
			(unsigned long long)compile.generated, (unsigned long long)compile.translated);
		writer.F("Metal persistent translation cache hits: %llu\n", (unsigned long long)compile.translationCacheHits);
		writer.F("Metal frontend GLSL/SPIR-V/MSL phase totals: %.3f/%.3f/%.3f ms\n",
			compile.phases.glslSeconds * 1000.0, compile.phases.spirvSeconds * 1000.0, compile.phases.mslSeconds * 1000.0);
		writer.F("Metal MSAA samples: %d\n", 1 << msaaLevel_);
		const auto &transforms = drawEngine_.TransformMetrics();
		writer.F("Metal flat draws/eligible/hardware/vertices: %llu/%llu/%llu/%llu; preparation: %.3f ms\n",
			(unsigned long long)transforms.flatDraws, (unsigned long long)transforms.flatHardwareEligible,
			(unsigned long long)transforms.flatHardwareDraws, (unsigned long long)transforms.flatVertices,
			transforms.flatPreparationSeconds * 1000.0);
		writer.F("Metal flat lighting fallbacks: %llu\n", (unsigned long long)transforms.flatLightingFallbacks);
		const auto regions = static_cast<TextureCacheMetal *>(textureCache_)->GetRegionMetrics();
		writer.F("Metal atlas totals hits/misses/draws/evictions/clears: %llu/%llu/%llu/%llu/%llu\n",
			(unsigned long long)regions.hits, (unsigned long long)regions.misses, (unsigned long long)regions.draws,
			(unsigned long long)regions.evictions, (unsigned long long)regions.clears);
		writer.F("Metal atlas adjacent region groups: %llu\n", (unsigned long long)regions.groups);
		writer.F("Metal atlas source/region bytes, entries: %zu/%zu, %zu\n", regions.sourceBytes, regions.regionBytes, regions.count);
		if (draw_) {
			writer.F("Metal render passes begun this frame: %u\n", draw_->GetCurrentBackendState().passes);
			auto *context = (Metal::RenderContext *)draw_->GetNativeObject(Draw::NativeObject::CONTEXT);
			if (context) {
				const auto metrics = context->Metrics();
				writer.F("Metal queue barriers/full/init boundaries: %llu/%llu/%llu; resource/prior-submission dependencies: %llu/%llu\n",
					(unsigned long long)metrics.queueBarriers, (unsigned long long)metrics.fullQueueBarriers,
					(unsigned long long)metrics.initializationBoundaries, (unsigned long long)metrics.resourceDependencies,
					(unsigned long long)metrics.priorSubmissionDependencies);
				writer.F("Metal CPU completion wait elapsed: %.3f ms (%llu calls, %llu pending at lock); not GPU barrier time\n",
					metrics.completionWaitSeconds * 1000.0, (unsigned long long)metrics.completionWaits,
					(unsigned long long)metrics.blockingCompletionWaits);
				writer.F("Metal drawable queue waits/presentations: %llu/%llu\n",
					(unsigned long long)metrics.drawableWaits, (unsigned long long)metrics.presentedDrawables);
				writer.F("Metal upload total payload/padding bytes: %llu/%llu; capacity: %zu; argument tables: %zu\n",
					(unsigned long long)metrics.uploadBytes, (unsigned long long)metrics.uploadPadding, metrics.uploadCapacity, metrics.argumentCapacity);
				writer.F("Metal library total CPU time: %.3f ms, %llu compiles\n", metrics.librarySeconds * 1000.0, (unsigned long long)metrics.libraryCompiles);
				writer.F("Metal transient copy-plane/sample-layer/backbuffer allocations: %llu/%llu/%llu; bytes: %llu/%llu/%llu\n",
					(unsigned long long)metrics.transientAllocations[0], (unsigned long long)metrics.transientAllocations[1], (unsigned long long)metrics.transientAllocations[2],
					(unsigned long long)metrics.transientBytes[0], (unsigned long long)metrics.transientBytes[1], (unsigned long long)metrics.transientBytes[2]);
				writer.F("Metal copy plane requests/capacity: %llu/%zu bytes\n", (unsigned long long)metrics.copyPlaneRequests, metrics.copyPlaneCapacity);
				const double gpuTimeMs = context->LastSubmissionGPUTimeMs();
				if (gpuTimeMs >= 0.0) {
					writer.F("Last Metal submission GPU time: %.3f ms\n", gpuTimeMs);
				}
			}
		}
	}

	bool GetCurrentFramebuffer(GPUDebugBuffer &buffer, GPUDebugFramebufferType type, int maxRes) override {
		return GPUCommonHWMetal::GetCurrentFramebuffer(buffer, type, maxRes) && ValidDebugImage(buffer);
	}
	bool GetCurrentDepthbuffer(GPUDebugBuffer &buffer) override {
		return GPUCommonHWMetal::GetCurrentDepthbuffer(buffer) && ValidDebugImage(buffer);
	}
	bool GetCurrentStencilbuffer(GPUDebugBuffer &buffer) override {
		return GPUCommonHWMetal::GetCurrentStencilbuffer(buffer) && ValidDebugImage(buffer);
	}
	bool GetOutputFramebuffer(GPUDebugBuffer &buffer) override {
		return GPUCommonHWMetal::GetOutputFramebuffer(buffer) && ValidDebugImage(buffer);
	}
	bool GetCurrentTexture(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) override {
		return GPUCommonHWMetal::GetCurrentTexture(buffer, level, isFramebuffer) && ValidDebugImage(buffer);
	}
	bool GetCurrentClut(GPUDebugBuffer &buffer) override {
		return GPUCommonHWMetal::GetCurrentClut(buffer) && ValidDebugImage(buffer);
	}

	void BuildReportingInfo() override {
		id<MTLDevice> device = (__bridge id<MTLDevice>)(void *)draw_->GetNativeObject(Draw::NativeObject::DEVICE);
		if (!device) {
			GPUCommonHWMetal::BuildReportingInfo();
			return;
		}
		reportingPrimaryInfo_ = device.name.UTF8String;
		reportingFullInfo_ = "Metal 4, MSL 4.0 (" + reportingPrimaryInfo_;
		if ([device supportsFamily:MTLGPUFamilyApple7]) {
			reportingFullInfo_ += ", Apple7+";
		} else if ([device supportsFamily:MTLGPUFamilyApple5]) {
			reportingFullInfo_ += ", Apple5+";
		} else if ([device supportsFamily:MTLGPUFamilyMac2]) {
			reportingFullInfo_ += ", Mac2";
		}
		if (@available(macOS 13.0, iOS 16.4, *)) {
			if (device.supportsBCTextureCompression) {
				reportingFullInfo_ += ", BC textures";
			}
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
		GPUCommonHWMetal::DeviceLost();
	}
	void DeviceRestore(Draw::DrawContext *draw) override {
		GPUCommonHWMetal::DeviceRestore(draw);
		metalShaders_->LoadCache(shaderCachePath_);
		drawEngine_.LoadPipelineCache(pipelineCachePath_);
	}

private:
	static bool ValidDebugImage(const GPUDebugBuffer &buffer) {
		return buffer.GetData() && buffer.GetStride() != 0 && buffer.GetHeight() != 0;
	}
	void BeginHostFrame(const DisplayLayoutConfig &config) override {
		GPUCommonHWMetal::BeginHostFrame(config);
		textureCache_->StartFrame();
		drawEngine_.BeginFrame();
		framebufferManager_->BeginFrame(config);
		if (gstate_c.useFlagsChanged) {
			shaderManager_->ClearShaders();
			framebufferManager_->ClearAllDepthBuffers();
			gstate_c.useFlagsChanged = false;
		}
		const int flips = gpuStats.totals.numFlips;
		const bool newFlip = flips != lastCacheCheckFlip_;
		lastCacheCheckFlip_ = flips;
		// Paused/slow PSP output can have several host frames at the same flip.
		if (newFlip && shaderCachePath_.Valid() && flips && !(flips & 32767)) {
			drawEngine_.SavePipelineCache(pipelineCachePath_);
			metalShaders_->SaveCache(shaderCachePath_);
		}
	}

	DrawEngineMetal drawEngine_;
	ShaderManagerMetal *metalShaders_ = nullptr;
	Path shaderCachePath_;
	Path pipelineCachePath_;
	int lastCacheCheckFlip_ = -1;
};

GPUCommon *CreateMetalGPU(GraphicsContext *context, Draw::DrawContext *draw) {
	return new GPU_Metal(context, draw);
}
