// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#if __has_include(<MetalFX/MetalFX.h>)
#import <MetalFX/MetalFX.h>
#define PPSSPP_HAS_METALFX 1
#else
#define PPSSPP_HAS_METALFX 0
#endif
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "Common/Data/Convert/ColorConv.h"
#include "Common/GPU/Metal/MetalResources.h"
#include "Common/GPU/Metal/MetalRenderManager.h"
#include "Common/GPU/Metal/thin3d_metal.h"
#include "Common/Log.h"
#include "Core/Config.h"
#include "GPU/GPUState.h"

namespace Draw {
namespace {

template<class T>
void BindRef(AutoRef<T> &ref, T *value) {
	if (ref.ptr != value) {
		ref = value;
	}
}

struct MetalBlend final : BlendState { BlendStateDesc desc; };
struct MetalDepth final : DepthStencilState { DepthStencilStateDesc desc; };
struct MetalRaster final : RasterState { RasterStateDesc desc; };
struct MetalSampler final : SamplerState { id<MTLSamplerState> state = nil; };
struct MetalInput final : InputLayout {
	MTLVertexDescriptor *desc = nil;
	int stride = 0;
};
struct MetalShader final : ShaderModule {
	ShaderStage stage;
	id<MTLFunction> function = nil;
	Metal::CompiledShader compiled;
	ShaderStage GetStage() const override { return stage; }
};
struct MetalPipelineRequest {
	std::mutex mutex;
	std::condition_variable ready;
	id<MTLRenderPipelineState> state = nil;
	std::string error;
	bool complete = false;
};
struct MetalPipeline final : Pipeline {
	MTLRenderPipelineDescriptor *desc = nil;
	DepthStencilStateDesc depth{};
	RasterStateDesc raster{};
	MTLPrimitiveType primitive;
	int stride = 0;
	size_t uniformSize = 0;
	uint32_t textureMask = 0;
	uint32_t vertexTextureMask = 0;
	uint32_t fragmentTextureMask = 0;
	bool multiview = false;
	bool fan = false;
	// Attachment formats are part of the PSO identity. This pipeline object
	// already uniquely owns its shaders, vertex layout, blend, and topology.
	std::map<std::array<uint64_t, 3>, id<MTLRenderPipelineState>> states;
	std::map<std::array<uint64_t, 3>, std::shared_ptr<MetalPipelineRequest>> pending;
	std::map<uint32_t, id<MTLDepthStencilState>> depthStates;
};

struct MetalDelayedReadback {
	id<MTLTexture> source = nil;
	id<MTLBuffer> ready = nil;
	// Retains reusable storage when pendingCommands is nil.
	id<MTLBuffer> pending = nil;
	id<MTLCommandBuffer> pendingCommands = nil;
	size_t pitch = 0;
	uint64_t lastUse = 0;
};

static MTLBlendFactor BlendFactorNative(BlendFactor factor) {
	static const MTLBlendFactor factors[] = {
		MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor, MTLBlendFactorOneMinusSourceColor,
		MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor,
		MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha,
		MTLBlendFactorDestinationAlpha, MTLBlendFactorOneMinusDestinationAlpha,
		MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
		MTLBlendFactorBlendAlpha, MTLBlendFactorOneMinusBlendAlpha,
		MTLBlendFactorSource1Color, MTLBlendFactorOneMinusSource1Color,
		MTLBlendFactorSource1Alpha, MTLBlendFactorOneMinusSource1Alpha,
	};
	return factors[(size_t)factor];
}

class MetalDrawContext final : public DrawContext, public Metal::RenderManager {
public:
	bool Init(std::string *error);
	~MetalDrawContext() override { Wait(); DestroyPresets(); }
	void Wait() override;
	const DeviceCaps &GetDeviceCaps() const override { return caps_; }
	uint32_t GetDataFormatSupport(DataFormat fmt) const override { return Metal::FormatSupport(context_.Device(), fmt); }
	uint32_t GetSupportedShaderLanguages() const override { return GLSL_VULKAN; }
	void SetErrorCallback(ErrorCallbackFn callback, void *userdata) override { errorCallback_ = callback; errorUserdata_ = userdata; }
	DepthStencilState *CreateDepthStencilState(const DepthStencilStateDesc &desc) override;
	BlendState *CreateBlendState(const BlendStateDesc &desc) override;
	SamplerState *CreateSamplerState(const SamplerStateDesc &desc) override;
	RasterState *CreateRasterState(const RasterStateDesc &desc) override;
	InputLayout *CreateInputLayout(const InputLayoutDesc &desc) override;
	ShaderModule *CreateShaderModule(ShaderStage stage, ShaderLanguage language, const uint8_t *data, size_t size, const char *tag) override;
	Pipeline *CreateGraphicsPipeline(const PipelineDesc &desc, const char *tag) override;
	Buffer *CreateBuffer(size_t size, uint32_t usage) override { return Metal::Buffer::Create(context_.Device(), size, usage); }
	Texture *CreateTexture(const TextureDesc &desc) override;
	Framebuffer *CreateFramebuffer(const FramebufferDesc &desc) override;
	void UpdateBuffer(Buffer *buffer, const uint8_t *data, size_t offset, size_t size, UpdateBufferFlags flags) override;
	void UpdateTextureLevels(Texture *texture, const uint8_t **data, TextureCallback callback, int levels) override;
	void UpdateTextureRegions(Texture *texture, int level, const TextureRegionUpdate *regions, int numRegions) override;
	void CopyFramebufferImage(Framebuffer *src, int level, int x, int y, int z, Framebuffer *dst, int dstLevel, int dstX, int dstY, int dstZ, int width, int height, int depth, Aspect aspects, const char *tag) override;
	bool BlitFramebuffer(Framebuffer *src, int sx1, int sy1, int sx2, int sy2, Framebuffer *dst, int dx1, int dy1, int dx2, int dy2, Aspect aspects, FBBlitFilter filter, const char *tag) override;
	bool CopyFramebufferToMemory(Framebuffer *src, Aspect aspect, int x, int y, int w, int h, DataFormat format, void *pixels, int stride, ReadbackMode mode, const char *tag) override;
	DataFormat PreferredFramebufferReadbackFormat(Framebuffer *src) override;
	void BindFramebufferAsRenderTarget(Framebuffer *fbo, const RenderPassInfo &rp, const char *tag) override;
	void BindFramebufferAsTexture(Framebuffer *fbo, int binding, Aspect aspect, int layer) override;
	bool SupportsSpatialUpscaling() const override { return spatialSupported_; }
	bool UpscaleBoundTexture(int binding, int width, int height, UVRect *uv) override;
	void GetFramebufferDimensions(Framebuffer *fbo, int *w, int *h) override;
	void SetScissorRect(int x, int y, int w, int h) override { scissor_ = {x, y, w, h}; }
	void SetViewport(const Viewport &viewport) override { viewport_ = viewport; }
	void SetBlendFactor(float color[4]) override { std::copy(color, color + 4, blendColor_.begin()); }
	void SetStencilParams(uint8_t ref, uint8_t write, uint8_t compare) override { stencilRef_ = ref; stencilWrite_ = write; stencilCompare_ = compare; }
	void BindSamplerStates(int start, int count, SamplerState **states) override;
	void BindTextures(int start, int count, Texture **textures, TextureBindFlags flags) override;
	void BindVertexBuffer(Buffer *buffer, int offset) override { BindRef(vertex_, static_cast<Metal::Buffer *>(buffer)); vertexOffset_ = offset; }
	void BindIndexBuffer(Buffer *buffer, int offset) override { BindRef(index_, static_cast<Metal::Buffer *>(buffer)); indexOffset_ = offset; }
	void BindNativeTexture(int slot, void *texture) override;
	void UpdateDynamicUniformBuffer(const void *data, size_t size) override;
	void Invalidate(InvalidationFlags flags) override;
	void BindPipeline(Pipeline *pipeline) override;
	void Draw(int count, int offset) override;
	void DrawIndexed(int count, int offset) override;
	void DrawUP(const void *data, int count) override;
	void DrawIndexedUP(const void *data, int count, const void *indices, int indexCount) override;
	void DrawIndexedClippedBatchUP(const void *data, int count, const void *indices, int indexCount, Slice<ClippedDraw> draws, const void *uniforms, size_t size) override;
	void BeginFrame(DebugFlags flags) override;
	std::string GetGpuProfileString() const override { return context_.GPUProfileString(); }
	void EndFrame() override { EndPass(); Invalidate(InvalidationFlags::CACHED_RENDER_STATE); }
	void Present(PresentMode mode) override;
	PresentMode GetCurrentPresentMode() const override { return presentMode_; }
	void Clear(Aspect aspects, uint32_t color, float depth, int stencil) override;
	void InvalidateFramebuffer(FBInvalidationStage stage, Aspect aspects) override;
	std::string GetInfoString(InfoField info) const override;
	uint64_t GetNativeObject(NativeObject obj, void *src) override;
	void HandleEvent(Event event, int w, int h, void *p1, void *p2) override;
	void SetInvalidationCallback(InvalidationCallback callback) override { invalidation_ = callback; }
	int GetFrameCount() override { return frameCount_; }
	BackendState GetCurrentBackendState() const override { return {passCount_, encoder_ != nil}; }
	Metal::RenderContext &Context() override { return context_; }
	id<MTLRenderCommandEncoder> RenderEncoder() override {
		if (!BeginPass()) {
			return nil;
		}
		// The GE draw sets native state directly on this encoder.
		thin3DEncoderState_.valid = false;
		return encoder_;
	}
	uint64_t RenderStateSerial() const override { return renderStateSerial_; }
	void EndRenderPass() override { EndPass(); }
	void SetNativeSampler(int slot, id<MTLSamplerState> sampler) override;
	bool RestoreBackbufferTarget(std::string *error) override;
	bool SnapshotBackbufferColor(int slot, std::string *error) override;
	bool BindTextures(id<MTLRenderCommandEncoder> encoder, uint32_t mask, std::string *error) override;
	Framebuffer *RenderTarget() const override { return target_.ptr; }
	MTLPixelFormat ColorFormat() const override { return target_.ptr ? target_->Color().pixelFormat : MTLPixelFormatInvalid; }
	MTLPixelFormat DepthStencilFormat() const override { return target_.ptr && target_->DepthStencil() ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid; }
	int SampleCount() const override { return target_ ? target_->SampleCount() : 1; }
	bool SetSurface(CAMetalLayer *layer, std::string *error) override;
	void ResizeSurface() override;

private:
	struct Thin3DEncoderState {
		bool valid = false;
		uintptr_t pipeline = 0;
		uintptr_t depth = 0;
		uintptr_t uniform = 0;
		size_t uniformOffset = 0;
		NSUInteger stencil = 0;
		MTLCullMode cull = MTLCullModeNone;
		MTLWinding winding = MTLWindingCounterClockwise;
		MTLViewport viewport{};
		MTLScissorRect scissor{};
		std::array<float, 4> blendColor{};
		bool viewMaskBound = false;
	};
	bool Commands();
	void EndPass();
	bool BeginPass();
	void RequestPipeline();
	bool Apply(id<MTLBuffer> vertices, size_t offset);
	NSUInteger InstanceCount() const { return pipeline_->multiview ? target_->Layers() : 1; }
	void Error(const std::string &message);
	void ReleaseDrawable();
	Metal::Framebuffer *Resolve(Framebuffer *framebuffer);
	id<MTLTexture> NullTexture(bool array);
	bool Copy(Framebuffer *src, int sx, int sy, Framebuffer *dst, int dx, int dy, int w, int h, Aspect aspects);
	bool ReadbackDelayedColor(id<MTLTexture> source, int x, int y, int w, int h, DataFormat format,
		void *pixels, int stride, std::string *error);
	void DrawFan(id<MTLBuffer> vertices, size_t vertexOffset, const uint16_t *indices, int count, int firstVertex);
	Metal::RenderContext context_;
	Metal::FramebufferCopy framebufferCopy_;
	std::map<std::array<uintptr_t, 5>, MetalDelayedReadback> delayedReadbacks_;
	uint64_t delayedReadbackUse_ = 0;
	bool spatialSupported_ = false;
	std::array<uint64_t, 5> spatialKey_{};
	int spatialRetryFrame_ = 0;
#if PPSSPP_HAS_METALFX
	id<MTLFXSpatialScaler> spatialScaler_ API_AVAILABLE(macos(13.0), ios(16.0)) = nil;
#endif
	id<MTLTexture> spatialInput_ = nil;
	id<MTLTexture> spatialOutput_ = nil;
	id<MTLTexture> nullTexture_ = nil;
	id<MTLTexture> nullTextureArray_ = nil;
	DeviceCaps caps_{};
	id<MTLRenderCommandEncoder> encoder_ = nil;
	uint64_t renderStateSerial_ = 0;
	Thin3DEncoderState thin3DEncoderState_;
	AutoRef<Metal::Framebuffer> target_;
	AutoRef<Metal::Framebuffer> backbuffer_;
	CAMetalLayer *layer_ = nil;
	id<CAMetalDrawable> drawable_ = nil;
	id<MTLTexture> surfaceDepth_ = nil;
	bool attemptedDrawable_ = false;
	AutoRef<MetalPipeline> pipeline_;
	AutoRef<Metal::Buffer> vertex_, index_;
	int vertexOffset_ = 0, indexOffset_ = 0;
	std::array<id<MTLTexture>, MAX_TEXTURE_SLOTS> textures_{};
	std::array<id<MTLSamplerState>, MAX_TEXTURE_SLOTS> samplers_{};
	uint32_t vertexTextureMask_ = 0;
	uint32_t fragmentTextureMask_ = 0;
	// Identity only: do not retain a previous drawable or render target here.
	std::array<uintptr_t, MAX_TEXTURE_SLOTS> boundVertexTextures_{};
	std::array<uintptr_t, MAX_TEXTURE_SLOTS> boundVertexSamplers_{};
	std::array<uintptr_t, MAX_TEXTURE_SLOTS> boundFragmentTextures_{};
	std::array<uintptr_t, MAX_TEXTURE_SLOTS> boundFragmentSamplers_{};
	uint64_t boundFragmentSerial_ = 0;
	bool boundVertexValid_ = false;
	bool boundFragmentValid_ = false;
	std::vector<uint8_t> uniformData_;
	Metal::UploadSlice uniform_;
	uint64_t uniformGeneration_ = 0;
	std::array<float, 4> blendColor_{};
	std::array<int, 4> scissor_{};
	Viewport viewport_{};
	uint8_t stencilRef_ = 0, stencilWrite_ = 255, stencilCompare_ = 255;
	RenderPassInfo pass_{RPAction::KEEP, RPAction::KEEP, RPAction::KEEP, 0, 1.0f, 0, "Metal"};
	Aspect discardStoreAspects_ = Aspect::NO_BIT;
	int frameCount_ = 0;
	uint32_t passCount_ = 0;
	PresentMode presentMode_ = PresentMode::FIFO;
	InvalidationCallback invalidation_;
	ErrorCallbackFn errorCallback_ = nullptr;
	void *errorUserdata_ = nullptr;
};

bool MetalDrawContext::Init(std::string *error) {
	if (!context_.Init(error, std::clamp(g_Config.iInflightFrames, 1, 2))) {
		return false;
	}
	context_.SetBeginCommandsCallback([this]() {
		if (invalidation_) {
			invalidation_(InvalidationCallbackFlags::COMMAND_BUFFER_STATE);
		}
	});
	shaderLanguageDesc_.Init(GLSL_VULKAN);
	shaderLanguageDesc_.framebufferArrayTextures = true;
	const bool appleGPU = [context_.Device() supportsFamily:MTLGPUFamilyApple1];
	caps_.vendor = appleGPU ? GPUVendor::VENDOR_APPLE : GPUVendor::VENDOR_UNKNOWN;
	caps_.deviceName = context_.DeviceName();
	caps_.maxTextureSize = Metal::MaxTextureDimension(context_.Device());
	caps_.coordConvention = CoordConvention::Vulkan;
	caps_.preferredDepthBufferFormat = DataFormat::D32F_S8;
	caps_.fragmentShaderFullPrecisionFloat = true;
	caps_.fragmentShaderInt32Supported = true;
	caps_.fragmentShaderDepthWriteSupported = true;
	caps_.fragmentShaderStencilWriteSupported = true;
	caps_.blendMinMaxSupported = true;
	caps_.dualSourceBlend = true;
	caps_.depthClampSupported = true;
	caps_.maxClipDistances = 8;
	caps_.anisoSupported = true;
	caps_.samplerLodControl = true;
	caps_.setMaxFrameLatencySupported = true;
	caps_.framebufferCopySupported = true;
	caps_.framebufferFetchSupported = appleGPU;
	caps_.framebufferBlitSupported = true;
	caps_.framebufferDepthBlitSupported = true;
	caps_.framebufferStencilBlitSupported = true;
	caps_.framebufferDepthCopySupported = true;
	caps_.framebufferSeparateDepthCopySupported = true;
	caps_.textureDepthSupported = true;
	caps_.texture3DSupported = true;
	caps_.isTilingGPU = appleGPU;
	caps_.textureSwizzleSupported = true;
	if (@available(macOS 13.0, iOS 16.0, *)) {
		caps_.multiViewSupported = [context_.Device() supportsFamily:MTLGPUFamilyApple5] ||
			[context_.Device() supportsFamily:MTLGPUFamilyMac2];
	}
	caps_.multiSampleLevelsMask = 1;
	if (Metal::SupportsDepthStencilResolve(context_.Device())) {
		for (int level = 1; level <= 4; ++level) {
			if ([context_.Device() supportsTextureSampleCount:1u << level]) {
				caps_.multiSampleLevelsMask |= 1u << level;
			}
		}
	}
	caps_.presentModesSupported = PresentMode::FIFO;
	caps_.presentMaxInterval = 1;
#if PPSSPP_PLATFORM(MAC)
	caps_.presentModesSupported |= PresentMode::IMMEDIATE;
	caps_.presentInstantModeChange = true;
#endif
#if PPSSPP_HAS_METALFX
	if (@available(macOS 13.0, iOS 16.0, *)) {
		spatialSupported_ = [MTLFXSpatialScalerDescriptor supportsDevice:context_.Device()];
	}
#endif
	return true;
}

bool MetalDrawContext::UpscaleBoundTexture(int binding, int width, int height, UVRect *uv) {
#if !PPSSPP_HAS_METALFX
	return false;
#else
	if (!spatialSupported_ || binding < 0 || binding >= MAX_TEXTURE_SLOTS || width <= 0 || height <= 0 ||
		width > caps_.maxTextureSize || height > caps_.maxTextureSize) {
		return false;
	}
	id<MTLTexture> source = textures_[binding];
	if (!source || source == spatialOutput_ || source.textureType != MTLTextureType2D) {
		return false;
	}
	const UVRect crop = uv ? *uv : UVRect{0.0f, 0.0f, 1.0f, 1.0f};
	const float left = std::min(crop.u0, crop.u1), right = std::max(crop.u0, crop.u1);
	const float top = std::min(crop.v0, crop.v1), bottom = std::max(crop.v0, crop.v1);
	if (!std::isfinite(crop.u0) || !std::isfinite(crop.u1) || !std::isfinite(crop.v0) || !std::isfinite(crop.v1) ||
		left < 0.0f || top < 0.0f || right > 1.0f || bottom > 1.0f || left >= right || top >= bottom) {
		return false;
	}
	const int sx = (int)floorf(left * source.width), sy = (int)floorf(top * source.height);
	const int sw = (int)ceilf(right * source.width) - sx, sh = (int)ceilf(bottom * source.height) - sy;
	if (sw <= 0 || sh <= 0) {
		return false;
	}
	const UVRect resultUV{(crop.u0 * source.width - sx) / sw, (crop.v0 * source.height - sy) / sh,
		(crop.u1 * source.width - sx) / sw, (crop.v1 * source.height - sy) / sh};
	// Fractional texel crops retain their exact UVs within the enclosing texels.
	const float outputWidth = ceilf(width / fabsf(resultUV.u1 - resultUV.u0));
	const float outputHeight = ceilf(height / fabsf(resultUV.v1 - resultUV.v0));
	if (!std::isfinite(outputWidth) || !std::isfinite(outputHeight) || outputWidth > caps_.maxTextureSize || outputHeight > caps_.maxTextureSize) {
		return false;
	}
	width = (int)outputWidth;
	height = (int)outputHeight;
	if (width < sw || height < sh || (width == sw && height == sh)) {
		return false;
	}
	if (source.pixelFormat != MTLPixelFormatRGBA8Unorm && source.pixelFormat != MTLPixelFormatBGRA8Unorm) {
		return false;
	}
	if (@available(macOS 13.0, iOS 16.0, *)) {
		const std::array<uint64_t, 5> key{(uint64_t)sw, (uint64_t)sh, (uint64_t)width, (uint64_t)height, source.pixelFormat};
		if (key != spatialKey_ || (!spatialScaler_ && frameCount_ >= spatialRetryFrame_)) {
			spatialKey_ = key;
			spatialScaler_ = nil;
			spatialInput_ = nil;
			spatialOutput_ = nil;
			MTLFXSpatialScalerDescriptor *desc = [MTLFXSpatialScalerDescriptor new];
			desc.inputWidth = sw;
			desc.inputHeight = sh;
			desc.outputWidth = width;
			desc.outputHeight = height;
			desc.colorTextureFormat = source.pixelFormat;
			desc.outputTextureFormat = source.pixelFormat;
			desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
			spatialScaler_ = [desc newSpatialScalerWithDevice:context_.Device()];
			if (spatialScaler_) {
				MTLTextureDescriptor *textureDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
					width:sw height:sh mipmapped:NO];
				textureDesc.storageMode = MTLStorageModePrivate;
				textureDesc.usage = spatialScaler_.colorTextureUsage;
				spatialInput_ = [context_.Device() newTextureWithDescriptor:textureDesc];
				textureDesc.width = width;
				textureDesc.height = height;
				textureDesc.usage = spatialScaler_.outputTextureUsage | MTLTextureUsageShaderRead;
				spatialOutput_ = [context_.Device() newTextureWithDescriptor:textureDesc];
			}
			if (!spatialScaler_ || !spatialInput_ || !spatialOutput_) {
				// Resource pressure can be temporary. Retry without allocating every frame.
				WARN_LOG(Log::G3D, "MetalFX Spatial could not allocate %dx%d output; using normal presentation", width, height);
				spatialScaler_ = nil;
				spatialInput_ = nil;
				spatialOutput_ = nil;
				spatialRetryFrame_ = frameCount_ + 120;
			}
		}
		if (!spatialScaler_ || !Commands()) {
			return false;
		}
		EndPass();
		std::string error;
		// Use a private input with MetalFX's required usage flags. The source may
		// be a resolved framebuffer or uploaded texture with different usage.
		if (!Metal::CopyImage(context_, source, sx, sy, spatialInput_, 0, 0, sw, sh, &error)) {
			Error(error);
			return false;
		}
		spatialScaler_.colorTexture = spatialInput_;
		spatialScaler_.outputTexture = spatialOutput_;
		spatialScaler_.inputContentWidth = sw;
		spatialScaler_.inputContentHeight = sh;
		[spatialScaler_ encodeToCommandBuffer:context_.Commands()];
		textures_[binding] = spatialOutput_;
		if (uv) {
			*uv = resultUV;
		}
		return true;
	}
	return false;
#endif
}

void MetalDrawContext::Error(const std::string &message) {
	ERROR_LOG(Log::G3D, "Metal: %s", message.c_str());
	if (errorCallback_) {
		errorCallback_("Metal rendering error", message.c_str(), errorUserdata_);
	}
}

bool MetalDrawContext::Commands() {
	if (context_.Commands()) {
		return true;
	}
	std::string error;
	if (!context_.BeginCommands(&error)) {
		Error(error);
		return false;
	}
	return true;
}

void MetalDrawContext::EndPass() {
	if (encoder_) {
		const bool multisample = target_ && target_->MultiSampleLevel() > 0;
		[encoder_ setColorStoreAction:(discardStoreAspects_ & Aspect::COLOR_BIT) ? MTLStoreActionDontCare :
			multisample ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore atIndex:0];
		if (target_ && target_->DepthStencil()) {
			const bool resolveDepthStencil = multisample;
			[encoder_ setDepthStoreAction:(discardStoreAspects_ & Aspect::DEPTH_BIT) ? MTLStoreActionDontCare :
				resolveDepthStencil ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore];
			[encoder_ setStencilStoreAction:(discardStoreAspects_ & Aspect::STENCIL_BIT) ? MTLStoreActionDontCare :
				resolveDepthStencil ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore];
		}
		[encoder_ endEncoding];
		encoder_ = nil;
		thin3DEncoderState_.valid = false;
	}
	discardStoreAspects_ = Aspect::NO_BIT;
}

void MetalDrawContext::Wait() {
	EndPass();
	std::string error;
	if (context_.Commands() && !context_.SubmitCommands(true, &error)) {
		Error(error);
	}
	if (!context_.WaitUntilIdle(&error)) {
		Error(error);
	}
}

DepthStencilState *MetalDrawContext::CreateDepthStencilState(const DepthStencilStateDesc &desc) {
	auto state = new MetalDepth();
	state->desc = desc;
	return state;
}

BlendState *MetalDrawContext::CreateBlendState(const BlendStateDesc &desc) {
	if (desc.logicEnabled) {
		Error("Logic operations require the shader fallback on Metal");
		return nullptr;
	}
	auto state = new MetalBlend();
	state->desc = desc;
	return state;
}

RasterState *MetalDrawContext::CreateRasterState(const RasterStateDesc &desc) {
	auto state = new MetalRaster();
	state->desc = desc;
	return state;
}

SamplerState *MetalDrawContext::CreateSamplerState(const SamplerStateDesc &desc) {
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	sd.minFilter = (MTLSamplerMinMagFilter)desc.minFilter;
	sd.magFilter = (MTLSamplerMinMagFilter)desc.magFilter;
	sd.mipFilter = desc.mipFilter == TextureFilter::NEAREST ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
	sd.maxAnisotropy = (NSUInteger)std::clamp(desc.maxAniso, 1.0f, 16.0f);
	const MTLSamplerAddressMode modes[] = {MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat, MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToBorderColor};
	sd.sAddressMode = modes[(size_t)desc.wrapU];
	sd.tAddressMode = modes[(size_t)desc.wrapV];
	sd.rAddressMode = modes[(size_t)desc.wrapW];
	sd.borderColor = desc.borderColor == OPAQUE_WHITE ? MTLSamplerBorderColorOpaqueWhite : desc.borderColor == OPAQUE_BLACK ? MTLSamplerBorderColorOpaqueBlack : MTLSamplerBorderColorTransparentBlack;
	sd.compareFunction = desc.shadowCompareEnabled ? (MTLCompareFunction)desc.shadowCompareFunc : MTLCompareFunctionNever;
	id<MTLSamplerState> native = [context_.Device() newSamplerStateWithDescriptor:sd];
	if (!native) {
		Error("Failed to create Metal sampler");
		return nullptr;
	}
	auto state = new MetalSampler();
	state->state = native;
	return state;
}

InputLayout *MetalDrawContext::CreateInputLayout(const InputLayoutDesc &desc) {
	if (desc.stride <= 0 || desc.stride > 2048) {
		return nullptr;
	}
	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	std::array<bool, 31> used{};
	for (const auto &a : desc.attributes) {
		const auto format = Metal::VertexFormat(a.format);
		if (a.location < 0 || a.location >= (int)used.size() || used[a.location] ||
			a.offset < 0 || a.offset > desc.stride || DataFormatSizeInBytes(a.format) > (size_t)(desc.stride - a.offset) || format == MTLVertexFormatInvalid) {
			Error("Invalid Metal vertex attribute layout");
			return nullptr;
		}
		used[a.location] = true;
		vd.attributes[a.location].format = format;
		vd.attributes[a.location].offset = a.offset;
		vd.attributes[a.location].bufferIndex = Metal::VERTEX_BUFFER_SLOT;
	}
	vd.layouts[Metal::VERTEX_BUFFER_SLOT].stride = desc.stride;
	vd.layouts[Metal::VERTEX_BUFFER_SLOT].stepFunction = MTLVertexStepFunctionPerVertex;
	auto layout = new MetalInput();
	layout->desc = vd;
	layout->stride = desc.stride;
	return layout;
}

ShaderModule *MetalDrawContext::CreateShaderModule(ShaderStage stage, ShaderLanguage language, const uint8_t *data, size_t size, const char *tag) {
	if (language != GLSL_VULKAN || stage == ShaderStage::Compute || !data || !size) {
		return nullptr;
	}
	Metal::CompiledShader compiled;
	std::string error;
	Metal::ShaderCompileOptions options;
	options.multiview = gstate_c.Use(GPU_USE_SINGLE_PASS_STEREO);
#if PPSSPP_PLATFORM(IOS)
	options.ios = true;
#endif
	if (!Metal::CompileShader(std::string_view((const char *)data, size), stage, options, &compiled, &error)) {
		Error(error);
		return nullptr;
	}
	for (const auto &resource : compiled.resources) {
		if ((resource.kind == Metal::ResourceKind::UniformBuffer && resource.binding != 0) ||
			(resource.kind == Metal::ResourceKind::SampledTexture && resource.index >= MAX_TEXTURE_SLOTS) ||
			resource.kind == Metal::ResourceKind::StorageBuffer || resource.kind == Metal::ResourceKind::StorageTexture) {
			Error("Shader resources exceed the thin3d Metal binding contract");
			return nullptr;
		}
	}
	id<MTLFunction> function = context_.CreateShader(compiled, tag, &error);
	if (!function) {
		Error(error);
		return nullptr;
	}
	auto shader = new MetalShader();
	shader->stage = stage;
	shader->function = function;
	shader->compiled = std::move(compiled);
	return shader;
}

Pipeline *MetalDrawContext::CreateGraphicsPipeline(const PipelineDesc &desc, const char *tag) {
	if (!desc.blend || !desc.depthStencil || !desc.raster || desc.prim > Primitive::TRIANGLE_FAN) {
		Error("Incomplete or unsupported Metal pipeline");
		return nullptr;
	}
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	size_t requiredUniforms = 0;
	uint32_t textureMask = 0;
	uint32_t vertexTextureMask = 0;
	uint32_t fragmentTextureMask = 0;
	bool multiview = false;
	for (auto module : desc.shaders) {
		if (!module) {
			Error("Metal pipeline has a null shader module");
			return nullptr;
		}
		auto shader = static_cast<MetalShader *>(module);
		multiview |= shader->compiled.needsViewMaskBuffer;
		if (shader->stage == ShaderStage::Vertex) {
			pd.vertexFunction = shader->function;
		} else if (shader->stage == ShaderStage::Fragment) {
			pd.fragmentFunction = shader->function;
		}
		for (const auto &resource : shader->compiled.resources) {
			if (resource.kind == Metal::ResourceKind::UniformBuffer) {
				requiredUniforms = std::max(requiredUniforms, (size_t)resource.byteSize);
			} else if (resource.kind == Metal::ResourceKind::SampledTexture) {
				const uint32_t bit = 1u << resource.index;
				textureMask |= bit;
				if (shader->stage == ShaderStage::Vertex) {
					vertexTextureMask |= bit;
				} else if (shader->stage == ShaderStage::Fragment) {
					fragmentTextureMask |= bit;
				}
			}
		}
	}
	if (!pd.vertexFunction || !pd.fragmentFunction ||
		(requiredUniforms && (!desc.uniformDesc || desc.uniformDesc->uniformBufferSize < requiredUniforms))) {
		Error("Metal pipeline shaders or uniform layout are incomplete");
		return nullptr;
	}
	if (tag) {
		pd.label = [NSString stringWithUTF8String:tag];
	}
	if (desc.inputLayout) {
		pd.vertexDescriptor = static_cast<MetalInput *>(desc.inputLayout)->desc;
	}
	pd.inputPrimitiveTopology = desc.prim == Primitive::POINT_LIST ? MTLPrimitiveTopologyClassPoint :
		(desc.prim == Primitive::LINE_LIST || desc.prim == Primitive::LINE_STRIP) ? MTLPrimitiveTopologyClassLine :
		MTLPrimitiveTopologyClassTriangle;
	const auto &blend = static_cast<MetalBlend *>(desc.blend)->desc;
	auto color = pd.colorAttachments[0];
	color.blendingEnabled = blend.enabled;
	color.sourceRGBBlendFactor = BlendFactorNative(blend.srcCol);
	color.destinationRGBBlendFactor = BlendFactorNative(blend.dstCol);
	color.sourceAlphaBlendFactor = BlendFactorNative(blend.srcAlpha);
	color.destinationAlphaBlendFactor = BlendFactorNative(blend.dstAlpha);
	color.rgbBlendOperation = (MTLBlendOperation)blend.eqCol;
	color.alphaBlendOperation = (MTLBlendOperation)blend.eqAlpha;
	color.writeMask = ((blend.colorMask & COLOR_MASK_R) ? MTLColorWriteMaskRed : 0) |
		((blend.colorMask & COLOR_MASK_G) ? MTLColorWriteMaskGreen : 0) |
		((blend.colorMask & COLOR_MASK_B) ? MTLColorWriteMaskBlue : 0) |
		((blend.colorMask & COLOR_MASK_A) ? MTLColorWriteMaskAlpha : 0);
	auto pipeline = new MetalPipeline();
	pipeline->desc = pd;
	pipeline->depth = static_cast<MetalDepth *>(desc.depthStencil)->desc;
	pipeline->raster = static_cast<MetalRaster *>(desc.raster)->desc;
	pipeline->stride = desc.inputLayout ? static_cast<MetalInput *>(desc.inputLayout)->stride : 0;
	pipeline->uniformSize = desc.uniformDesc ? desc.uniformDesc->uniformBufferSize : 0;
	pipeline->textureMask = textureMask;
	pipeline->vertexTextureMask = vertexTextureMask;
	pipeline->fragmentTextureMask = fragmentTextureMask;
	pipeline->multiview = multiview;
	pipeline->fan = desc.prim == Primitive::TRIANGLE_FAN;
	const MTLPrimitiveType types[] = {MTLPrimitiveTypePoint, MTLPrimitiveTypeLine, MTLPrimitiveTypeLineStrip, MTLPrimitiveTypeTriangle, MTLPrimitiveTypeTriangleStrip};
	pipeline->primitive = pipeline->fan ? MTLPrimitiveTypeTriangle : types[(size_t)desc.prim];
	return pipeline;
}

void MetalDrawContext::BindPipeline(Pipeline *pipeline) {
	BindRef(pipeline_, static_cast<MetalPipeline *>(pipeline));
	RequestPipeline();
}

void MetalDrawContext::RequestPipeline() {
	if (!pipeline_ || !target_) {
		return;
	}
	const auto colorFormat = target_->Color().pixelFormat;
	const auto depthFormat = target_->DepthStencil() ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid;
	const std::array<uint64_t, 3> key{(uint64_t)colorFormat, (uint64_t)depthFormat, (uint64_t)target_->SampleCount()};
	if (pipeline_->states.find(key) != pipeline_->states.end() || pipeline_->pending.find(key) != pipeline_->pending.end()) {
		return;
	}
	MTLRenderPipelineDescriptor *desc = [pipeline_->desc copy];
	desc.colorAttachments[0].pixelFormat = colorFormat;
	desc.depthAttachmentPixelFormat = depthFormat;
	desc.stencilAttachmentPixelFormat = depthFormat;
	desc.rasterSampleCount = target_->SampleCount();
	auto request = std::make_shared<MetalPipelineRequest>();
	pipeline_->pending.emplace(key, request);
	[context_.Device() newRenderPipelineStateWithDescriptor:desc completionHandler:^(id<MTLRenderPipelineState> state, NSError *nativeError) {
		@autoreleasepool {
			std::lock_guard<std::mutex> lock(request->mutex);
			request->state = state;
			if (!state) {
				const char *description = nativeError.localizedDescription.UTF8String;
				request->error = description ? description : "Metal pipeline creation failed";
			}
			request->complete = true;
		}
		request->ready.notify_all();
	}];
}

Texture *MetalDrawContext::CreateTexture(const TextureDesc &desc) {
	// New allocations are initialized before this render buffer is submitted.
	// Preserve the render pass across unrelated texture initializations.
	std::string error;
	auto texture = Metal::Texture::Create(context_, desc, &error);
	if (!texture) {
		Error(error);
	}
	return texture;
}

Framebuffer *MetalDrawContext::CreateFramebuffer(const FramebufferDesc &desc) {
	std::string error;
	auto fbo = Metal::Framebuffer::Create(context_.Device(), desc, &error);
	if (!fbo) {
		Error(error);
	}
	return fbo;
}

void MetalDrawContext::UpdateBuffer(Buffer *buffer, const uint8_t *data, size_t offset, size_t size, UpdateBufferFlags flags) {
	if (!buffer || !static_cast<Metal::Buffer *>(buffer)->Update(context_.Device(), data, offset, size, flags)) {
		Error("Metal buffer update failed or exceeded allocation");
	}
}

void MetalDrawContext::UpdateTextureLevels(Texture *texture, const uint8_t **data, TextureCallback callback, int levels) {
	EndPass();
	std::string error;
	if (!texture || !static_cast<Metal::Texture *>(texture)->Update(context_, data, callback, levels, &error)) {
		Error(error.empty() ? "No Metal texture to update" : error);
	}
}

void MetalDrawContext::UpdateTextureRegions(Texture *texture, int level, const TextureRegionUpdate *regions, int numRegions) {
	if (numRegions <= 0) {
		return;
	}
	EndPass();
	std::string error;
	if (!texture || !static_cast<Metal::Texture *>(texture)->UpdateRegions(context_, level, regions, numRegions, &error)) {
		Error(error.empty() ? "No Metal texture to update" : error);
	}
}

Metal::Framebuffer *MetalDrawContext::Resolve(Framebuffer *fbo) {
	if (fbo) {
		return static_cast<Metal::Framebuffer *>(fbo);
	}
	if (layer_) {
		if (backbuffer_) {
			return backbuffer_.ptr;
		}
		// Don't repeatedly block waiting for a hidden/unavailable surface during
		// the same frame. Offscreen GE work can still be encoded and submitted.
		if (attemptedDrawable_ || targetWidth_ <= 0 || targetHeight_ <= 0) {
			return nullptr;
		}
		attemptedDrawable_ = true;
		@autoreleasepool {
			drawable_ = [layer_ nextDrawable];
			if (!drawable_) {
				return nullptr;
			}
			id<MTLTexture> color = drawable_.texture;
			SetTargetSize((int)color.width, (int)color.height);
			if (!surfaceDepth_ || surfaceDepth_.width != color.width || surfaceDepth_.height != color.height) {
				MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:color.width height:color.height mipmapped:NO];
				desc.storageMode = MTLStorageModePrivate;
				desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				surfaceDepth_ = [context_.Device() newTextureWithDescriptor:desc];
				if (!surfaceDepth_) {
					Error("Failed to allocate Metal surface depth/stencil");
					drawable_ = nil;
					return nullptr;
				}
			}
			std::string error;
			backbuffer_.reset(Metal::Framebuffer::Wrap(color, surfaceDepth_, &error));
			if (!backbuffer_) {
				Error(error);
				drawable_ = nil;
			}
		}
		return backbuffer_.ptr;
	}
	if (!backbuffer_ || backbuffer_->Width() != targetWidth_ || backbuffer_->Height() != targetHeight_) {
		backbuffer_.reset(static_cast<Metal::Framebuffer *>(CreateFramebuffer({targetWidth_, targetHeight_, 1, 1, 0, true, "Metal offscreen output"})));
	}
	return backbuffer_.ptr;
}

void MetalDrawContext::GetFramebufferDimensions(Framebuffer *fbo, int *w, int *h) {
	*w = fbo ? fbo->Width() : targetWidth_;
	*h = fbo ? fbo->Height() : targetHeight_;
}

void MetalDrawContext::BindFramebufferAsRenderTarget(Framebuffer *fbo, const RenderPassInfo &rp, const char *tag) {
	// A KEEP rebind of the active target does not need another MSAA resolve.
	// Keep the split when a bound texture aliases the target or a store discard is pending.
	if (encoder_ && fbo && target_.ptr == fbo &&
		rp.color == RPAction::KEEP && rp.depth == RPAction::KEEP && rp.stencil == RPAction::KEEP &&
		discardStoreAspects_ == Aspect::NO_BIT) {
		bool readsTarget = false;
		for (id<MTLTexture> texture : textures_) {
			if (texture && target_->OwnsTexture(texture)) {
				readsTarget = true;
				break;
			}
		}
		if (!readsTarget) {
			return;
		}
	}
	EndPass();
	BindRef(target_, Resolve(fbo));
	pass_ = rp;
	pass_.tag = tag;
	discardStoreAspects_ = Aspect::NO_BIT;
	RequestPipeline();
	// Execute a clear even when this pass has no draws or is immediately rebound.
	if (rp.color == RPAction::CLEAR || rp.depth == RPAction::CLEAR || rp.stencil == RPAction::CLEAR) {
		BeginPass();
	} else if (target_ && invalidation_) {
		// GE state must be dirtied when the target changes, even if encoding waits for a draw.
		invalidation_(InvalidationCallbackFlags::RENDER_PASS_STATE);
	}
}

bool MetalDrawContext::BeginPass() {
	if (encoder_) {
		return true;
	}
	if (!target_ || !Commands()) {
		return false;
	}
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	const MTLLoadAction actions[] = {MTLLoadActionLoad, MTLLoadActionClear, MTLLoadActionDontCare};
	target_->SetRenderAttachments(rp);
	rp.colorAttachments[0].storeAction = MTLStoreActionUnknown;
	if (target_->DepthStencil()) {
		rp.depthAttachment.storeAction = MTLStoreActionUnknown;
		rp.stencilAttachment.storeAction = MTLStoreActionUnknown;
	}
	rp.colorAttachments[0].loadAction = actions[(size_t)pass_.color];
	const uint32_t c = pass_.clearColor;
	rp.colorAttachments[0].clearColor = MTLClearColorMake((c & 255) / 255.0, ((c >> 8) & 255) / 255.0, ((c >> 16) & 255) / 255.0, (c >> 24) / 255.0);
	if (target_->DepthStencil()) {
		rp.depthAttachment.loadAction = actions[(size_t)pass_.depth];
		rp.depthAttachment.clearDepth = pass_.clearDepth;
		rp.stencilAttachment.loadAction = actions[(size_t)pass_.stencil];
		rp.stencilAttachment.clearStencil = pass_.clearStencil;
	}
	context_.ProfileRenderPass(rp, pass_.tag);
	encoder_ = [context_.Commands() renderCommandEncoderWithDescriptor:rp];
	if (!encoder_) {
		Error("Failed to begin Metal render pass");
		return false;
	}
	++renderStateSerial_;
	thin3DEncoderState_.valid = false;
	vertexTextureMask_ = 0;
	fragmentTextureMask_ = 0;
	boundVertexTextures_.fill(0);
	boundVertexSamplers_.fill(0);
	boundFragmentTextures_.fill(0);
	boundFragmentSamplers_.fill(0);
	boundVertexValid_ = false;
	boundFragmentValid_ = false;
	// A transfer/readback may split this logical pass. Its continuation must
	// load the stored contents, never replay the original clear/don't-care.
	pass_.color = pass_.depth = pass_.stencil = RPAction::KEEP;
	++passCount_;
	if (invalidation_) {
		invalidation_(InvalidationCallbackFlags::RENDER_PASS_STATE);
	}
	return true;
}

void MetalDrawContext::BindFramebufferAsTexture(Framebuffer *fbo, int binding, Aspect aspect, int layer) {
	if (binding < 0 || binding >= (int)MAX_TEXTURE_SLOTS || (layer < 0 && layer != ALL_LAYERS)) {
		Error("Invalid Metal framebuffer texture binding");
		return;
	}
	auto source = Resolve(fbo);
	if (!source || (aspect != Aspect::COLOR_BIT && aspect != Aspect::DEPTH_BIT) || layer >= source->Layers()) {
		textures_[binding] = nil;
		Error("Metal framebuffer texture binding requires color or depth");
		return;
	}
	if (layer == ALL_LAYERS) {
		textures_[binding] = aspect == Aspect::DEPTH_BIT ?
			(shaderLanguageDesc_.framebufferArrayTextures ? source->DepthStencilArray() : source->DepthStencil()) :
			(shaderLanguageDesc_.framebufferArrayTextures ? source->ColorArray() : source->Color());
	} else {
		textures_[binding] = aspect == Aspect::DEPTH_BIT ? source->DepthStencilLayer(layer) : source->ColorLayer(layer);
	}
	// A different MSAA target was resolved when its own pass ended. Only the
	// current target needs its active pass closed before sampling its resolve.
	if (source->MultiSampleLevel() > 0 && target_.ptr == source) {
		EndPass();
	}
	if (!textures_[binding]) {
		Error("Missing Metal framebuffer texture attachment");
	}
}

bool MetalDrawContext::Copy(Framebuffer *src, int sx, int sy, Framebuffer *dst, int dx, int dy, int w, int h, Aspect aspects) {
	EndPass();
	auto source = Resolve(src);
	auto dest = Resolve(dst);
	if (!source || !dest) {
		return false;
	}
	std::string error;
	if (dest->MultiSampleLevel() > 0) {
		if (!framebufferCopy_.Copy(context_, source, sx, sy, dest, dx, dy, w, h, aspects, &error)) {
			Error(error);
			return false;
		}
		return true;
	}
	const bool depth = aspects & Aspect::DEPTH_BIT;
	const bool stencil = aspects & Aspect::STENCIL_BIT;
	if ((depth || stencil) && (!source->DepthStencil() || !dest->DepthStencil())) {
		Error("Metal depth/stencil copy requires depth/stencil attachments");
		return false;
	}
	if ((aspects & Aspect::COLOR_BIT) && !Metal::CopyImage(context_, source->Color(), sx, sy, dest->Color(), dx, dy, w, h, &error)) {
		Error(error);
		return false;
	}
	if ((depth || stencil) && !Metal::CopyDepthStencil(context_, source->DepthStencil(), sx, sy,
		dest->DepthStencil(), dx, dy, w, h,
		depth && stencil ? Aspect::DEPTH_BIT | Aspect::STENCIL_BIT : depth ? Aspect::DEPTH_BIT : Aspect::STENCIL_BIT, &error)) {
		Error(error);
		return false;
	}
	return true;
}

void MetalDrawContext::CopyFramebufferImage(Framebuffer *src, int level, int x, int y, int z, Framebuffer *dst, int dstLevel, int dx, int dy, int dz, int w, int h, int depth, Aspect aspects, const char *tag) {
	if (level || dstLevel || z || dz || depth != 1) {
		Error("Unsupported Metal framebuffer copy mip or layer");
		return;
	}
	Copy(src, x, y, dst, dx, dy, w, h, aspects);
}

bool MetalDrawContext::BlitFramebuffer(Framebuffer *src, int sx1, int sy1, int sx2, int sy2, Framebuffer *dst, int dx1, int dy1, int dx2, int dy2, Aspect aspects, FBBlitFilter filter, const char *tag) {
	if (sx2 <= sx1 || sy2 <= sy1 || dx2 <= dx1 || dy2 <= dy1) {
		return false;
	}
	if (sx2 - sx1 == dx2 - dx1 && sy2 - sy1 == dy2 - dy1 && filter == FB_BLIT_NEAREST) {
		return Copy(src, sx1, sy1, dst, dx1, dy1, sx2 - sx1, sy2 - sy1, aspects);
	}
	EndPass();
	auto source = Resolve(src);
	auto dest = Resolve(dst);
	if (!source || !dest) {
		return false;
	}
	std::string error;
	if (!framebufferCopy_.Blit(context_, source, sx1, sy1, sx2 - sx1, sy2 - sy1,
		dest, dx1, dy1, dx2 - dx1, dy2 - dy1, aspects, filter, &error)) {
		Error(error);
		return false;
	}
	return true;
}

bool MetalDrawContext::ReadbackDelayedColor(id<MTLTexture> source, int x, int y, int w, int h, DataFormat format,
	void *pixels, int stride, std::string *error) {
	error->clear();
	if (!pixels || stride < 0 || !source || source.sampleCount != 1 ||
		(source.textureType != MTLTextureType2D && source.textureType != MTLTextureType2DArray) ||
		(source.pixelFormat != MTLPixelFormatRGBA8Unorm && source.pixelFormat != MTLPixelFormatBGRA8Unorm) ||
		(format != DataFormat::R8G8B8A8_UNORM && format != DataFormat::B8G8R8A8_UNORM &&
			format != DataFormat::R8G8B8_UNORM && format != DataFormat::R5G6B5_UNORM_PACK16 &&
			format != DataFormat::A1R5G5B5_UNORM_PACK16 && format != DataFormat::A4R4G4B4_UNORM_PACK16) ||
		x < 0 || y < 0 || w <= 0 || h <= 0 || (uint64_t)x + w > source.width || (uint64_t)y + h > source.height) {
		*error = "Invalid delayed Metal color readback";
		return false;
	}
	const std::array<uintptr_t, 5> key{(uintptr_t)(__bridge void *)source, (uintptr_t)x, (uintptr_t)y, (uintptr_t)w, (uintptr_t)h};
	auto found = delayedReadbacks_.find(key);
	if (found == delayedReadbacks_.end() && delayedReadbacks_.size() >= 8) {
		auto victim = delayedReadbacks_.end();
		for (auto it = delayedReadbacks_.begin(); it != delayedReadbacks_.end(); ++it) {
			const auto commands = it->second.pendingCommands;
			if (commands && commands.status != MTLCommandBufferStatusCompleted && commands.status != MTLCommandBufferStatusError) {
				continue;
			}
			if (victim == delayedReadbacks_.end() || it->second.lastUse < victim->second.lastUse) {
				victim = it;
			}
		}
		if (victim == delayedReadbacks_.end()) {
			// Keep the cache bounded even when many different targets are read in
			// one frame. A blocking readback is still correct for this rare case.
			return Metal::Readback(context_, source, Aspect::COLOR_BIT, x, y, w, h, format, pixels, stride, error);
		}
		delayedReadbacks_.erase(victim);
	}
	auto &readback = delayedReadbacks_[key];
	readback.source = source;
	readback.lastUse = ++delayedReadbackUse_;
	if (readback.pendingCommands) {
		if (readback.pendingCommands.status == MTLCommandBufferStatusCompleted) {
			std::swap(readback.ready, readback.pending);
			readback.pendingCommands = nil;
		} else if (readback.pendingCommands.status == MTLCommandBufferStatusError) {
			readback.pendingCommands = nil;
		}
	}
	if (!readback.pendingCommands) {
		const size_t pitch = ((size_t)w * 4 + 255) & ~size_t(255);
		if (pitch > context_.Device().maxBufferLength / h || !Commands()) {
			*error = "Failed to prepare delayed Metal readback";
			return false;
		}
		id<MTLBuffer> buffer = readback.pending;
		if (!buffer) {
			buffer = [context_.Device() newBufferWithLength:pitch * h options:MTLResourceStorageModeShared];
		}
		if (!buffer) {
			*error = "Failed to allocate delayed Metal readback buffer";
			return false;
		}
		id<MTLBlitCommandEncoder> blit = context_.BlitEncoder(context_.Commands(), "Delayed framebuffer readback");
		if (!blit) {
			*error = "Failed to encode delayed Metal readback";
			return false;
		}
		[blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x, y, 0)
			sourceSize:MTLSizeMake(w, h, 1) toBuffer:buffer destinationOffset:0
			destinationBytesPerRow:pitch destinationBytesPerImage:pitch * h options:MTLBlitOptionNone];
		[blit endEncoding];
		readback.pending = buffer;
		readback.pendingCommands = context_.Commands();
		readback.pitch = pitch;
	}
	if (!readback.ready) {
		return false;
	}
	if (source.pixelFormat == MTLPixelFormatBGRA8Unorm) {
		ConvertFromBGRA8888((uint8_t *)pixels, (const uint8_t *)readback.ready.contents, stride, (uint32_t)readback.pitch / 4, w, h, format);
	} else if (format == DataFormat::B8G8R8A8_UNORM) {
		for (int row = 0; row < h; ++row) {
			auto dst = (uint32_t *)pixels + row * stride;
			auto src = (const uint32_t *)((const uint8_t *)readback.ready.contents + row * readback.pitch);
			ConvertRGBA8888ToBGRA8888(dst, src, w);
		}
	} else {
		ConvertFromRGBA8888((uint8_t *)pixels, (const uint8_t *)readback.ready.contents, stride, (uint32_t)readback.pitch / 4, w, h, format);
	}
	return true;
}

bool MetalDrawContext::CopyFramebufferToMemory(Framebuffer *src, Aspect aspect, int x, int y, int w, int h, DataFormat format, void *pixels, int stride, ReadbackMode mode, const char *tag) {
	EndPass();
	auto fbo = Resolve(src);
	if (!fbo) {
		return false;
	}
	std::string error;
	const bool success = mode == ReadbackMode::OLD_DATA_OK && aspect == Aspect::COLOR_BIT ?
		ReadbackDelayedColor(fbo->Color(), x, y, w, h, format, pixels, stride, &error) :
		Metal::Readback(context_, aspect == Aspect::COLOR_BIT ? fbo->Color() : fbo->DepthStencil(), aspect, x, y, w, h, format, pixels, stride, &error);
	if (!success && !error.empty()) {
		Error(std::string(tag ? tag : "framebuffer readback") + ": " + error);
	}
	return success;
}

DataFormat MetalDrawContext::PreferredFramebufferReadbackFormat(Framebuffer *src) {
	if (!src && backbuffer_ && backbuffer_->Color().pixelFormat == MTLPixelFormatBGRA8Unorm) {
		return DataFormat::B8G8R8A8_UNORM;
	}
	return DrawContext::PreferredFramebufferReadbackFormat(src);
}

void MetalDrawContext::BindSamplerStates(int start, int count, SamplerState **states) {
	if (start < 0 || count < 0 || start > (int)MAX_TEXTURE_SLOTS - count) {
		Error("Invalid Metal sampler range");
		return;
	}
	for (int i = 0; i < count; ++i) {
		samplers_[start + i] = states[i] ? static_cast<MetalSampler *>(states[i])->state : nil;
	}
}

id<MTLTexture> MetalDrawContext::NullTexture(bool array) {
	if (!nullTexture_) {
		MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
			width:1 height:1 mipmapped:NO];
		desc.storageMode = MTLStorageModeShared;
		desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		nullTexture_ = [context_.Device() newTextureWithDescriptor:desc];
		if (!nullTexture_) {
			Error("Failed to create Metal null texture");
			return nil;
		}
		const uint32_t transparentBlack = 0;
		[nullTexture_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0
			withBytes:&transparentBlack bytesPerRow:sizeof(transparentBlack)];
	}
	if (array && !nullTextureArray_) {
		nullTextureArray_ = [nullTexture_ newTextureViewWithPixelFormat:nullTexture_.pixelFormat
			textureType:MTLTextureType2DArray levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
		if (!nullTextureArray_) {
			Error("Failed to create Metal null texture array view");
		}
	}
	return array ? nullTextureArray_ : nullTexture_;
}

void MetalDrawContext::BindTextures(int start, int count, Texture **textures, TextureBindFlags flags) {
	if (start < 0 || count < 0 || start > (int)MAX_TEXTURE_SLOTS - count) {
		Error("Invalid Metal texture range");
		return;
	}
	for (int i = 0; i < count; ++i) {
		auto *texture = textures[i] ? static_cast<Metal::Texture *>(textures[i]) : nullptr;
		textures_[start + i] = texture ?
			((flags & TextureBindFlags::VULKAN_BIND_ARRAY) ? texture->ArrayView() : texture->Native()) :
			NullTexture(flags & TextureBindFlags::VULKAN_BIND_ARRAY);
		if (texture && !textures_[start + i]) {
			Error("Failed to create Metal texture array view");
		}
	}
}

void MetalDrawContext::BindNativeTexture(int slot, void *texture) {
	if (slot >= 0 && slot < (int)MAX_TEXTURE_SLOTS) {
		textures_[slot] = (__bridge id<MTLTexture>)texture;
	}
}

void MetalDrawContext::SetNativeSampler(int slot, id<MTLSamplerState> sampler) {
	if (slot >= 0 && slot < (int)MAX_TEXTURE_SLOTS) {
		samplers_[slot] = sampler;
	}
}

bool MetalDrawContext::RestoreBackbufferTarget(std::string *error) {
	if (!backbuffer_) {
		*error = "Metal unbuffered GE draw has no backbuffer";
		return false;
	}
	if (target_.ptr != backbuffer_.ptr) {
		BindFramebufferAsRenderTarget(backbuffer_.ptr, {RPAction::KEEP, RPAction::KEEP, RPAction::KEEP}, "Metal unbuffered GE target restore");
	}
	return true;
}

bool MetalDrawContext::SnapshotBackbufferColor(int slot, std::string *error) {
	if (slot < 0 || slot >= (int)MAX_TEXTURE_SLOTS || !backbuffer_ || target_.ptr != backbuffer_.ptr) {
		*error = "Metal shader blending requires the bound backbuffer";
		return false;
	}
	id<MTLTexture> source = backbuffer_->Color();
	if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1) {
		*error = "Metal shader blending requires a single-sample backbuffer";
		return false;
	}
	EndPass();
	MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
		width:source.width height:source.height mipmapped:NO];
	desc.storageMode = MTLStorageModePrivate;
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	id<MTLTexture> snapshot = [context_.Device() newTextureWithDescriptor:desc];
	if (!snapshot) {
		*error = "Failed to allocate Metal backbuffer snapshot";
		return false;
	}
	if (!Metal::CopyImage(context_, source, 0, 0, snapshot, 0, 0, (int)source.width, (int)source.height, error)) {
		return false;
	}
	textures_[slot] = [snapshot newTextureViewWithPixelFormat:snapshot.pixelFormat textureType:MTLTextureType2DArray
		levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
	if (!textures_[slot]) {
		*error = "Failed to create Metal backbuffer snapshot array view";
		return false;
	}
	return true;
}

bool MetalDrawContext::BindTextures(id<MTLRenderCommandEncoder> encoder, uint32_t mask, std::string *error) {
	error->clear();
	if (!encoder || encoder != encoder_ || !target_ || (mask >> MAX_TEXTURE_SLOTS)) {
		*error = "Invalid Metal GE texture bindings or render encoder";
		return false;
	}
	for (int i = 0; i < (int)MAX_TEXTURE_SLOTS; ++i) {
		if (!(mask & (1u << i))) {
			continue;
		}
		if (!textures_[i] || !samplers_[i]) {
			*error = "Missing Metal GE texture or sampler at slot " + std::to_string(i) +
				" (texture=" + (textures_[i] ? "yes" : "no") + ", sampler=" + (samplers_[i] ? "yes" : "no") + ")";
			return false;
		}
		if (target_->OwnsTexture(textures_[i])) {
			*error = "Metal GE framebuffer feedback requires a separate copy";
			return false;
		}
	}
	const uint32_t slotsToUpdate = mask | fragmentTextureMask_;
	const bool rebind = !boundFragmentValid_ || boundFragmentSerial_ != renderStateSerial_;
	if (rebind) {
		boundFragmentTextures_.fill(0);
		boundFragmentSamplers_.fill(0);
	}
	for (int i = 0; i < (int)MAX_TEXTURE_SLOTS; ++i) {
		if (!(slotsToUpdate & (1u << i))) {
			continue;
		}
		id<MTLTexture> texture = (mask & (1u << i)) ? textures_[i] : nil;
		id<MTLSamplerState> sampler = (mask & (1u << i)) ? samplers_[i] : nil;
		const uintptr_t textureID = (uintptr_t)(__bridge void *)texture;
		const uintptr_t samplerID = (uintptr_t)(__bridge void *)sampler;
		if (rebind || boundFragmentTextures_[i] != textureID) {
			[encoder setFragmentTexture:texture atIndex:i];
		}
		if (rebind || boundFragmentSamplers_[i] != samplerID) {
			[encoder setFragmentSamplerState:sampler atIndex:i];
		}
		boundFragmentTextures_[i] = textureID;
		boundFragmentSamplers_[i] = samplerID;
	}
	boundFragmentSerial_ = renderStateSerial_;
	boundFragmentValid_ = true;
	fragmentTextureMask_ = mask;
	return true;
}

void MetalDrawContext::UpdateDynamicUniformBuffer(const void *data, size_t size) {
	uniform_ = {};
	uniformGeneration_ = 0;
	if (!data || !size) {
		uniformData_.clear();
		return;
	}
	if (!context_.Device() || size > context_.Device().maxBufferLength) {
		uniformData_.clear();
		Error("Invalid Metal dynamic uniform size or device");
		return;
	}
	uniformData_.assign((const uint8_t *)data, (const uint8_t *)data + size);
}

void MetalDrawContext::Invalidate(InvalidationFlags flags) {
	if (flags & InvalidationFlags::CACHED_RENDER_STATE) {
		thin3DEncoderState_.valid = false;
		pipeline_.reset(nullptr);
		textures_.fill(nil);
		samplers_.fill(nil);
	}
}

bool MetalDrawContext::Apply(id<MTLBuffer> vertices, size_t offset) {
	if (!target_ && layer_) {
		// A minimized/timed-out surface is an ordinary dropped presentation,
		// not a missing game resource or a fatal rendering error.
		return false;
	}
	if (!pipeline_ || !target_ || (pipeline_->stride && (!vertices || offset >= vertices.length)) ||
		(pipeline_->uniformSize && uniformData_.size() < pipeline_->uniformSize)) {
		Error("Incomplete Metal draw bindings");
		return false;
	}
	if (pipeline_->raster.cull == CullMode::FRONT_AND_BACK &&
		(pipeline_->primitive == MTLPrimitiveTypeTriangle || pipeline_->primitive == MTLPrimitiveTypeTriangleStrip)) {
		return false;
	}
	for (int i = 0; i < (int)MAX_TEXTURE_SLOTS; ++i) {
		if (!(pipeline_->textureMask & (1u << i))) {
			continue;
		}
		if (!textures_[i] || !samplers_[i]) {
			Error("Missing Metal shader texture or sampler binding");
			return false;
		}
		if (target_->OwnsTexture(textures_[i])) {
			Error("Metal framebuffer feedback requires a separate copy");
			return false;
		}
	}
	const int64_t right = (int64_t)scissor_[0] + scissor_[2];
	const int64_t bottom = (int64_t)scissor_[1] + scissor_[3];
	const int x = std::clamp(scissor_[0], 0, target_->Width());
	const int y = std::clamp(scissor_[1], 0, target_->Height());
	const int w = (int)std::clamp<int64_t>(right, x, target_->Width()) - x;
	const int h = (int)std::clamp<int64_t>(bottom, y, target_->Height()) - y;
	if (!w || !h || viewport_.Width <= 0 || viewport_.Height <= 0) {
		return false;
	}
	const auto colorFormat = target_->Color().pixelFormat;
	const auto depthFormat = target_->DepthStencil() ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid;
	const std::array<uint64_t, 3> key{(uint64_t)colorFormat, (uint64_t)depthFormat, (uint64_t)target_->SampleCount()};
	auto found = pipeline_->states.find(key);
	id<MTLRenderPipelineState> state = found == pipeline_->states.end() ? nil : found->second;
	if (!state) {
		RequestPipeline();
		auto pending = pipeline_->pending.find(key);
		if (pending == pipeline_->pending.end()) {
			Error("Metal pipeline request is unavailable");
			return false;
		}
		auto request = pending->second;
		std::unique_lock<std::mutex> lock(request->mutex);
		request->ready.wait(lock, [&] { return request->complete; });
		state = request->state;
		const std::string error = request->error;
		lock.unlock();
		pipeline_->pending.erase(pending);
		if (!state) {
			Error(error);
			return false;
		}
		pipeline_->states[key] = state;
	}
	const bool hasDepthStencil = target_->DepthStencil() != nil;
	const uint32_t depthKey = stencilWrite_ | (stencilCompare_ << 8) | (hasDepthStencil ? (1u << 16) : 0);
	id<MTLDepthStencilState> depthState = pipeline_->depthStates[depthKey];
	if (!depthState) {
		const auto &d = pipeline_->depth;
		MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
		dd.depthCompareFunction = hasDepthStencil && d.depthTestEnabled ? (MTLCompareFunction)d.depthCompare : MTLCompareFunctionAlways;
		dd.depthWriteEnabled = hasDepthStencil && d.depthWriteEnabled;
		if (hasDepthStencil && d.stencilEnabled) {
			MTLStencilDescriptor *s = [MTLStencilDescriptor new];
			s.stencilCompareFunction = (MTLCompareFunction)d.stencil.compareOp;
			s.stencilFailureOperation = (MTLStencilOperation)d.stencil.failOp;
			s.depthFailureOperation = (MTLStencilOperation)d.stencil.depthFailOp;
			s.depthStencilPassOperation = (MTLStencilOperation)d.stencil.passOp;
			s.readMask = stencilCompare_;
			s.writeMask = stencilWrite_;
			dd.frontFaceStencil = s;
			dd.backFaceStencil = s;
		}
		depthState = [context_.Device() newDepthStencilStateWithDescriptor:dd];
		if (!depthState) {
			Error("Failed to create Metal depth/stencil state");
			return false;
		}
		pipeline_->depthStates[depthKey] = depthState;
	}
	if (!BeginPass()) {
		return false;
	}
	if (!uniformData_.empty() && (!uniform_ || uniformGeneration_ != context_.CommandGeneration())) {
		std::string error;
		uniform_ = context_.Upload(uniformData_.data(), uniformData_.size(), &error);
		if (!uniform_) {
			Error(error);
			return false;
		}
		uniformGeneration_ = context_.CommandGeneration();
	}
	++renderStateSerial_;
	auto &bound = thin3DEncoderState_;
	const bool valid = bound.valid;
	const uintptr_t pipelineID = (uintptr_t)(__bridge void *)state;
	const uintptr_t depthID = (uintptr_t)(__bridge void *)depthState;
	const MTLCullMode cull = pipeline_->raster.cull == CullMode::BACK ? MTLCullModeBack :
		pipeline_->raster.cull == CullMode::FRONT ? MTLCullModeFront : MTLCullModeNone;
	const MTLWinding winding = pipeline_->raster.frontFace == Facing::CCW ? MTLWindingCounterClockwise : MTLWindingClockwise;
	const MTLViewport viewport{viewport_.TopLeftX, viewport_.TopLeftY, viewport_.Width, viewport_.Height, viewport_.MinDepth, viewport_.MaxDepth};
	const MTLScissorRect scissor{(NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h};
	if (!valid || bound.pipeline != pipelineID) {
		[encoder_ setRenderPipelineState:state];
	}
	if (!valid || bound.depth != depthID) {
		[encoder_ setDepthStencilState:depthState];
	}
	if (!valid || bound.stencil != stencilRef_) {
		[encoder_ setStencilReferenceValue:stencilRef_];
	}
	if (!valid || bound.cull != cull) {
		[encoder_ setCullMode:cull];
	}
	if (!valid || bound.winding != winding) {
		[encoder_ setFrontFacingWinding:winding];
	}
	// GE draws can enable depth clamping on this encoder; thin3d uses clipping.
	if (!valid) {
		[encoder_ setDepthClipMode:MTLDepthClipModeClip];
	}
	if (!valid || bound.viewport.originX != viewport.originX || bound.viewport.originY != viewport.originY ||
		bound.viewport.width != viewport.width || bound.viewport.height != viewport.height ||
		bound.viewport.znear != viewport.znear || bound.viewport.zfar != viewport.zfar) {
		[encoder_ setViewport:viewport];
	}
	if (!valid || bound.scissor.x != scissor.x || bound.scissor.y != scissor.y ||
		bound.scissor.width != scissor.width || bound.scissor.height != scissor.height) {
		[encoder_ setScissorRect:scissor];
	}
	if (!valid || bound.blendColor != blendColor_) {
		[encoder_ setBlendColorRed:blendColor_[0] green:blendColor_[1] blue:blendColor_[2] alpha:blendColor_[3]];
	}
	bound.pipeline = pipelineID;
	bound.depth = depthID;
	bound.stencil = stencilRef_;
	bound.cull = cull;
	bound.winding = winding;
	bound.viewport = viewport;
	bound.scissor = scissor;
	bound.blendColor = blendColor_;
	if (vertices) {
		[encoder_ setVertexBuffer:vertices offset:offset atIndex:Metal::VERTEX_BUFFER_SLOT];
	}
	if (uniform_) {
		const uintptr_t uniformID = (uintptr_t)(__bridge void *)uniform_.buffer;
		if (!valid || bound.uniform != uniformID || bound.uniformOffset != uniform_.offset) {
			[encoder_ setVertexBuffer:uniform_.buffer offset:uniform_.offset atIndex:0];
			[encoder_ setFragmentBuffer:uniform_.buffer offset:uniform_.offset atIndex:0];
		}
		bound.uniform = uniformID;
		bound.uniformOffset = uniform_.offset;
	}
	if (pipeline_->multiview && (!valid || !bound.viewMaskBound)) {
		const uint32_t viewMask[] = { 0, 2 };
		[encoder_ setVertexBytes:viewMask length:sizeof(viewMask) atIndex:Metal::VIEW_MASK_BUFFER_SLOT];
		[encoder_ setFragmentBytes:viewMask length:sizeof(viewMask) atIndex:Metal::VIEW_MASK_BUFFER_SLOT];
		bound.viewMaskBound = true;
	}
	const uint32_t vertexSlots = pipeline_->vertexTextureMask | vertexTextureMask_;
	const uint32_t fragmentSlots = pipeline_->fragmentTextureMask | fragmentTextureMask_;
	for (int i = 0; i < (int)MAX_TEXTURE_SLOTS; ++i) {
		const uint32_t bit = 1u << i;
		if (vertexSlots & bit) {
			const bool used = (pipeline_->vertexTextureMask & bit) != 0;
			id<MTLTexture> texture = used ? textures_[i] : nil;
			id<MTLSamplerState> sampler = used ? samplers_[i] : nil;
			const uintptr_t textureID = (uintptr_t)(__bridge void *)texture;
			const uintptr_t samplerID = (uintptr_t)(__bridge void *)sampler;
			if (!boundVertexValid_ || boundVertexTextures_[i] != textureID) {
				[encoder_ setVertexTexture:texture atIndex:i];
			}
			if (!boundVertexValid_ || boundVertexSamplers_[i] != samplerID) {
				[encoder_ setVertexSamplerState:sampler atIndex:i];
			}
			boundVertexTextures_[i] = textureID;
			boundVertexSamplers_[i] = samplerID;
		}
		if (fragmentSlots & bit) {
			const bool used = (pipeline_->fragmentTextureMask & bit) != 0;
			id<MTLTexture> texture = used ? textures_[i] : nil;
			id<MTLSamplerState> sampler = used ? samplers_[i] : nil;
			const uintptr_t textureID = (uintptr_t)(__bridge void *)texture;
			const uintptr_t samplerID = (uintptr_t)(__bridge void *)sampler;
			if (!boundFragmentValid_ || boundFragmentTextures_[i] != textureID) {
				[encoder_ setFragmentTexture:texture atIndex:i];
			}
			if (!boundFragmentValid_ || boundFragmentSamplers_[i] != samplerID) {
				[encoder_ setFragmentSamplerState:sampler atIndex:i];
			}
			boundFragmentTextures_[i] = textureID;
			boundFragmentSamplers_[i] = samplerID;
		}
	}
	vertexTextureMask_ = pipeline_->vertexTextureMask;
	fragmentTextureMask_ = pipeline_->fragmentTextureMask;
	boundVertexValid_ = true;
	boundFragmentSerial_ = renderStateSerial_;
	boundFragmentValid_ = true;
	bound.valid = true;
	return true;
}

void MetalDrawContext::DrawFan(id<MTLBuffer> vertices, size_t vertexOffset, const uint16_t *indices, int count, int firstVertex) {
	if (count < 3) {
		return;
	}
	const size_t indexCount = (size_t)(count - 2) * 3;
	if (indexCount > context_.Device().maxBufferLength / sizeof(uint32_t) || !Commands()) {
		Error("Metal triangle fan exceeds the index buffer limit");
		return;
	}
	std::string error;
	auto uploaded = context_.ReserveUpload(indexCount * sizeof(uint32_t), &error);
	if (!uploaded) {
		Error(error);
		return;
	}
	uint32_t *fan = (uint32_t *)((uint8_t *)uploaded.buffer.contents + uploaded.offset);
	const uint32_t center = indices ? indices[0] : (uint32_t)firstVertex;
	for (int i = 1; i < count - 1; ++i) {
		const size_t out = (size_t)(i - 1) * 3;
		fan[out] = center;
		fan[out + 1] = indices ? indices[i] : (uint32_t)firstVertex + (uint32_t)i;
		fan[out + 2] = indices ? indices[i + 1] : (uint32_t)firstVertex + (uint32_t)i + 1;
	}
	if (Apply(vertices, vertexOffset)) {
		[encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:indexCount indexType:MTLIndexTypeUInt32
			indexBuffer:uploaded.buffer indexBufferOffset:uploaded.offset instanceCount:InstanceCount()];
	}
}

void MetalDrawContext::Draw(int count, int offset) {
	if (count <= 0 || offset < 0 || vertexOffset_ < 0 || !pipeline_) {
		return;
	}
	if (pipeline_->stride && (!vertex_ || (uint64_t)vertexOffset_ + ((uint64_t)offset + count) * pipeline_->stride > vertex_->Size())) {
		Error("Metal draw exceeds vertex buffer");
		return;
	}
	if (!Commands()) {
		return;
	}
	std::string error;
	auto vertices = vertex_ ? vertex_->Snapshot(context_, &error) : Metal::UploadSlice{};
	if (vertex_ && !vertices) {
		Error(error);
		return;
	}
	const size_t vertexOffset = vertices.offset + (size_t)vertexOffset_;
	if (pipeline_->fan) {
		DrawFan(vertices.buffer, vertexOffset, nullptr, count, offset);
		return;
	}
	if (Apply(vertices.buffer, vertexOffset)) {
		[encoder_ drawPrimitives:pipeline_->primitive vertexStart:offset vertexCount:count instanceCount:InstanceCount()];
	}
}

void MetalDrawContext::DrawIndexed(int count, int offset) {
	if (count <= 0 || offset < 0 || indexOffset_ < 0 || vertexOffset_ < 0 || !index_ || !pipeline_) {
		return;
	}
	const size_t start = (size_t)indexOffset_ + (size_t)offset * 2;
	if ((start & 1) || start > index_->Size() || (size_t)count > (index_->Size() - start) / 2) {
		Error("Metal draw exceeds index buffer");
		return;
	}
	if (!Commands()) {
		return;
	}
	std::string error;
	auto vertices = vertex_ ? vertex_->Snapshot(context_, &error) : Metal::UploadSlice{};
	if (vertex_ && !vertices) {
		Error(error);
		return;
	}
	const size_t vertexOffset = vertices.offset + (size_t)vertexOffset_;
	if (pipeline_->fan) {
		if (!index_->Data()) {
			Error("Metal triangle fan indices are not CPU-readable");
			return;
		}
		DrawFan(vertices.buffer, vertexOffset, (const uint16_t *)(index_->Data() + start), count, 0);
		return;
	}
	auto indices = index_->Snapshot(context_, &error);
	if (!indices) {
		Error(error);
		return;
	}
	if (Apply(vertices.buffer, vertexOffset)) {
		[encoder_ drawIndexedPrimitives:pipeline_->primitive indexCount:count indexType:MTLIndexTypeUInt16
			indexBuffer:indices.buffer indexBufferOffset:indices.offset + start instanceCount:InstanceCount()];
	}
}

void MetalDrawContext::DrawUP(const void *data, int count) {
	if (!pipeline_ || count <= 0 || pipeline_->stride <= 0) {
		return;
	}
	if (!Commands()) {
		return;
	}
	std::string error;
	auto vertices = context_.Upload(data, (size_t)count * pipeline_->stride, &error);
	if (!vertices) {
		Error(error);
		return;
	}
	if (pipeline_->fan) {
		DrawFan(vertices.buffer, vertices.offset, nullptr, count, 0);
		return;
	}
	if (Apply(vertices.buffer, vertices.offset)) {
		[encoder_ drawPrimitives:pipeline_->primitive vertexStart:0 vertexCount:count instanceCount:InstanceCount()];
	}
}

void MetalDrawContext::DrawIndexedUP(const void *data, int count, const void *indices, int indexCount) {
	if (!pipeline_ || count <= 0 || !indices || indexCount <= 0 || pipeline_->stride <= 0) {
		return;
	}
	if (!Commands()) {
		return;
	}
	std::string error;
	auto vertices = context_.Upload(data, (size_t)count * pipeline_->stride, &error);
	if (!vertices) {
		Error(error);
		return;
	}
	if (pipeline_->fan) {
		DrawFan(vertices.buffer, vertices.offset, (const uint16_t *)indices, indexCount, 0);
		return;
	}
	auto index = context_.Upload(indices, (size_t)indexCount * 2, &error);
	if (!index) {
		Error(error);
		return;
	}
	if (Apply(vertices.buffer, vertices.offset)) {
		[encoder_ drawIndexedPrimitives:pipeline_->primitive indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:index.buffer indexBufferOffset:index.offset instanceCount:InstanceCount()];
	}
}

void MetalDrawContext::DrawIndexedClippedBatchUP(const void *data, int count, const void *indices, int indexCount, Slice<ClippedDraw> draws, const void *uniforms, size_t size) {
	UpdateDynamicUniformBuffer(uniforms, size);
	for (const auto &draw : draws) {
		if (draw.indexOffset < 0 || draw.indexCount <= 0 || draw.indexOffset > indexCount - draw.indexCount) {
			continue;
		}
		BindPipeline(draw.pipeline);
		SetScissorRect(draw.clipx, draw.clipy, draw.clipw, draw.cliph);
		SamplerState *sampler = draw.samplerState;
		BindSamplerStates(0, 1, &sampler);
		if (draw.bindFramebufferAsTex) {
			BindFramebufferAsTexture(draw.bindFramebufferAsTex, 0, draw.aspect, 0);
		} else if (draw.bindNativeTexture) {
			BindNativeTexture(0, draw.bindNativeTexture);
		} else {
			BindTexture(0, draw.bindTexture);
		}
		DrawIndexedUP(data, count, (const uint16_t *)indices + draw.indexOffset, draw.indexCount);
	}
}

void MetalDrawContext::BeginFrame(DebugFlags flags) {
	Present(presentMode_);
	const bool profileLog = (flags & DebugFlags::PROFILE_SCOPES) || g_Config.bGpuLogProfiler;
	context_.SetProfilingEnabled(profileLog || (flags & DebugFlags::PROFILE_TIMESTAMPS), profileLog);
	attemptedDrawable_ = false;
	passCount_ = 0;
	++frameCount_;
	Commands();
}

void MetalDrawContext::Present(PresentMode mode) {
	EndPass();
#if PPSSPP_PLATFORM(MAC)
	const PresentMode nextMode = mode == PresentMode::IMMEDIATE ? PresentMode::IMMEDIATE : PresentMode::FIFO;
	if (presentMode_ != nextMode) {
		presentMode_ = nextMode;
		if (layer_) {
			layer_.displaySyncEnabled = presentMode_ == PresentMode::FIFO;
		}
	}
#endif
	std::string error;
	// A readback may have submitted all render commands already. Present on a
	// fresh buffer in the same queue so it remains ordered after that work.
	if (drawable_ && Commands()) {
		[context_.Commands() presentDrawable:drawable_];
	}
	if (context_.Commands() && !context_.SubmitCommands(false, &error)) {
		Error(error);
	}
	if (layer_) {
		ReleaseDrawable();
	}
}

void MetalDrawContext::ReleaseDrawable() {
	// target_ can retain the surface even after rebinding it through a raw FBO
	// pointer. Never keep that texture bound after returning its drawable.
	if (target_.ptr == backbuffer_.ptr) {
		target_ = nullptr;
	}
	if (backbuffer_) {
		for (auto &texture : textures_) {
			if (backbuffer_->OwnsTexture(texture)) {
				texture = nil;
			}
		}
	}
	backbuffer_ = nullptr;
	drawable_ = nil;
}

bool MetalDrawContext::SetSurface(CAMetalLayer *layer, std::string *error) {
	error->clear();
	if (layer && ![layer isKindOfClass:[CAMetalLayer class]]) {
		*error = "Expected a CAMetalLayer";
		return false;
	}
	if (@available(macOS 13.0, iOS 16.0, *)) {
		Wait();
		ReleaseDrawable();
		surfaceDepth_ = nil;
		layer_ = layer;
		attemptedDrawable_ = false;
		if (layer_) {
			layer_.device = context_.Device();
			layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
			// Screenshots and framebuffer transfers can read the drawable.
			layer_.framebufferOnly = NO;
			layer_.maximumDrawableCount = std::clamp(g_Config.iInflightFrames + 1, 2, 3);
			layer_.allowsNextDrawableTimeout = YES;
#if PPSSPP_PLATFORM(MAC)
			layer_.displaySyncEnabled = presentMode_ == PresentMode::FIFO;
#endif
			ResizeSurface();
		}
		return true;
	}
	*error = "Metal 3 surfaces require macOS 13 or iOS 16";
	return false;
}

void MetalDrawContext::ResizeSurface() {
	if (!layer_) {
		return;
	}
	const CGSize size = layer_.drawableSize;
	const int w = (int)size.width;
	const int h = (int)size.height;
	if (w == targetWidth_ && h == targetHeight_) {
		return;
	}
	EndPass();
	ReleaseDrawable();
	surfaceDepth_ = nil;
	SetTargetSize(w, h);
	attemptedDrawable_ = false;
}

void MetalDrawContext::Clear(Aspect aspects, uint32_t color, float depth, int stencil) {
	if (aspects == Aspect::NO_BIT) {
		return;
	}
	EndPass();
	pass_ = {(aspects & Aspect::COLOR_BIT) ? RPAction::CLEAR : RPAction::KEEP,
		(aspects & Aspect::DEPTH_BIT) ? RPAction::CLEAR : RPAction::KEEP,
		(aspects & Aspect::STENCIL_BIT) ? RPAction::CLEAR : RPAction::KEEP,
		color, depth, (uint8_t)stencil, "Metal clear"};
	BeginPass();
}

void MetalDrawContext::InvalidateFramebuffer(FBInvalidationStage stage, Aspect aspects) {
	if (stage == FB_INVALIDATION_STORE) {
		discardStoreAspects_ |= aspects;
	} else if (stage == FB_INVALIDATION_LOAD) {
		if (encoder_) {
			discardStoreAspects_ |= aspects;
			EndPass();
		}
		if (aspects & Aspect::COLOR_BIT) {
			pass_.color = RPAction::DONT_CARE;
		}
		if (aspects & Aspect::DEPTH_BIT) {
			pass_.depth = RPAction::DONT_CARE;
		}
		if (aspects & Aspect::STENCIL_BIT) {
			pass_.stencil = RPAction::DONT_CARE;
		}
	}
}

std::string MetalDrawContext::GetInfoString(InfoField info) const {
	switch (info) {
	case InfoField::APINAME: return "Metal";
	case InfoField::APIVERSION: return "3";
	case InfoField::SHADELANGVERSION: return "MSL 3.0";
	case InfoField::VENDOR: return caps_.vendor == GPUVendor::VENDOR_APPLE ? "Apple" : "Unknown";
	case InfoField::VENDORSTRING: return context_.DeviceName();
	default: return "";
	}
}

uint64_t MetalDrawContext::GetNativeObject(NativeObject obj, void *src) {
	switch (obj) {
	case NativeObject::CONTEXT: return (uint64_t)&context_;
	case NativeObject::RENDER_MANAGER: return (uint64_t)static_cast<Metal::RenderManager *>(this);
	case NativeObject::DEVICE: return (uint64_t)(__bridge void *)context_.Device();
	case NativeObject::TEXTURE_VIEW: return src ? (uint64_t)(__bridge void *)static_cast<Metal::Texture *>(src)->Native() : 0;
	case NativeObject::NULL_IMAGEVIEW: return (uint64_t)(__bridge void *)NullTexture(false);
	case NativeObject::NULL_IMAGEVIEW_ARRAY: return (uint64_t)(__bridge void *)NullTexture(true);
	case NativeObject::BOUND_TEXTURE0_IMAGEVIEW: return (uint64_t)(__bridge void *)textures_[0];
	case NativeObject::BOUND_TEXTURE1_IMAGEVIEW: return (uint64_t)(__bridge void *)textures_[1];
	case NativeObject::BACKBUFFER_COLOR_TEX:
	case NativeObject::BACKBUFFER_COLOR_VIEW: return backbuffer_ ? (uint64_t)(__bridge void *)backbuffer_->Color() : 0;
	case NativeObject::BACKBUFFER_DEPTH_TEX:
	case NativeObject::BACKBUFFER_DEPTH_VIEW: return backbuffer_ ? (uint64_t)(__bridge void *)backbuffer_->DepthStencil() : 0;
	default: return 0;
	}
}

void MetalDrawContext::HandleEvent(Event event, int w, int h, void *p1, void *p2) {
	if (event == Event::RESIZED || event == Event::GOT_BACKBUFFER || event == Event::LOST_BACKBUFFER) {
		EndPass();
		target_ = nullptr;
		ReleaseDrawable();
		surfaceDepth_ = nil;
		attemptedDrawable_ = false;
		SetTargetSize(w, h);
	}
}

}  // namespace

DrawContext *T3DCreateMetalContext(std::string *error) {
	auto context = std::make_unique<MetalDrawContext>();
	if (!context->Init(error)) {
		return nullptr;
	}
	return context.release();
}

}  // namespace Draw
