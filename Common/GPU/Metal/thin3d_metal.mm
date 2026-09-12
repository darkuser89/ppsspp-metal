// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#import <MetalFX/MetalFX.h>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>

#include "Common/GPU/Metal/MetalResources.h"
#include "Common/GPU/Metal/MetalRenderManager.h"
#include "Common/GPU/Metal/thin3d_metal.h"
#include "Common/Log.h"

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
struct MetalPipeline final : Pipeline {
	MTLRenderPipelineDescriptor *desc = nil;
	DepthStencilStateDesc depth{};
	RasterStateDesc raster{};
	MTLPrimitiveType primitive;
	int stride = 0;
	size_t uniformSize = 0;
	uint32_t textureMask = 0;
	// Attachment formats are part of the PSO identity. This pipeline object
	// already uniquely owns its shaders, vertex layout, blend, and topology.
	std::map<std::array<uint64_t, 3>, id<MTLRenderPipelineState>> states;
	std::map<uint32_t, id<MTLDepthStencilState>> depthStates;
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
	Buffer *CreateBuffer(size_t size, uint32_t usage) override { return Metal::Buffer::Create(context_.Device(), size); }
	Texture *CreateTexture(const TextureDesc &desc) override;
	Framebuffer *CreateFramebuffer(const FramebufferDesc &desc) override;
	void UpdateBuffer(Buffer *buffer, const uint8_t *data, size_t offset, size_t size, UpdateBufferFlags flags) override;
	void UpdateTextureLevels(Texture *texture, const uint8_t **data, TextureCallback callback, int levels) override;
	void CopyFramebufferImage(Framebuffer *src, int level, int x, int y, int z, Framebuffer *dst, int dstLevel, int dstX, int dstY, int dstZ, int width, int height, int depth, Aspect aspects, const char *tag) override;
	bool BlitFramebuffer(Framebuffer *src, int sx1, int sy1, int sx2, int sy2, Framebuffer *dst, int dx1, int dy1, int dx2, int dy2, Aspect aspects, FBBlitFilter filter, const char *tag) override;
	bool CopyFramebufferToMemory(Framebuffer *src, Aspect aspect, int x, int y, int w, int h, DataFormat format, void *pixels, int stride, ReadbackMode mode, const char *tag) override;
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
	// Every Apply writes all dynamic state, including after native API calls.
	void Invalidate(InvalidationFlags flags) override {}
	void BindPipeline(Pipeline *pipeline) override { BindRef(pipeline_, static_cast<MetalPipeline *>(pipeline)); }
	void Draw(int count, int offset) override;
	void DrawIndexed(int count, int offset) override;
	void DrawUP(const void *data, int count) override;
	void DrawIndexedUP(const void *data, int count, const void *indices, int indexCount) override;
	void DrawIndexedClippedBatchUP(const void *data, int count, const void *indices, int indexCount, Slice<ClippedDraw> draws, const void *uniforms, size_t size) override;
	void BeginFrame(DebugFlags flags) override;
	void EndFrame() override { EndPass(); }
	void Present(PresentMode mode) override;
	PresentMode GetCurrentPresentMode() const override { return PresentMode::FIFO; }
	void Clear(Aspect aspects, uint32_t color, float depth, int stencil) override;
	std::string GetInfoString(InfoField info) const override;
	uint64_t GetNativeObject(NativeObject obj, void *src) override;
	void HandleEvent(Event event, int w, int h, void *p1, void *p2) override;
	void SetInvalidationCallback(InvalidationCallback callback) override { invalidation_ = callback; }
	int GetFrameCount() override { return frameCount_; }
	BackendState GetCurrentBackendState() const override { return {passCount_, encoder_ != nil}; }
	Metal::RenderContext &Context() override { return context_; }
	id<MTLRenderCommandEncoder> RenderEncoder() override { return BeginPass() ? encoder_ : nil; }
	void EndRenderPass() override { EndPass(); }
	void SetNativeSampler(int slot, id<MTLSamplerState> sampler) override;
	bool BindTextures(id<MTLRenderCommandEncoder> encoder, uint32_t mask, std::string *error) override;
	Framebuffer *RenderTarget() const override { return target_.ptr; }
	MTLPixelFormat ColorFormat() const override { return target_.ptr ? target_->Color().pixelFormat : MTLPixelFormatInvalid; }
	MTLPixelFormat DepthStencilFormat() const override { return target_.ptr && target_->DepthStencil() ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid; }
	int SampleCount() const override { return target_ ? target_->SampleCount() : 1; }
	bool SetSurface(CAMetalLayer *layer, std::string *error) override;
	void ResizeSurface() override;

private:
	bool Commands();
	void EndPass();
	bool BeginPass();
	bool Apply(id<MTLBuffer> vertices, size_t offset);
	void Error(const std::string &message);
	void ReleaseDrawable();
	Metal::Framebuffer *Resolve(Framebuffer *framebuffer);
	bool Copy(Framebuffer *src, int sx, int sy, Framebuffer *dst, int dx, int dy, int w, int h, Aspect aspects);
	id<MTLBuffer> Upload(const void *data, size_t size);
	Metal::RenderContext context_;
	Metal::FramebufferCopy framebufferCopy_;
	bool spatialSupported_ = false;
	std::array<uint64_t, 5> spatialKey_{};
	id<MTLFXSpatialScaler> spatialScaler_ = nil;
	id<MTLTexture> spatialInput_ = nil;
	id<MTLTexture> spatialOutput_ = nil;
	DeviceCaps caps_{};
	id<MTLRenderCommandEncoder> encoder_ = nil;
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
	id<MTLBuffer> uniform_ = nil;
	std::array<float, 4> blendColor_{};
	std::array<int, 4> scissor_{};
	Viewport viewport_{};
	uint8_t stencilRef_ = 0, stencilWrite_ = 255, stencilCompare_ = 255;
	RenderPassInfo pass_{RPAction::KEEP, RPAction::KEEP, RPAction::KEEP, 0, 1.0f, 0, "Metal"};
	int frameCount_ = 0;
	uint32_t passCount_ = 0;
	InvalidationCallback invalidation_;
	ErrorCallbackFn errorCallback_ = nullptr;
	void *errorUserdata_ = nullptr;
};

bool MetalDrawContext::Init(std::string *error) {
	if (!context_.Init(error)) {
		return false;
	}
	shaderLanguageDesc_.Init(GLSL_VULKAN);
	shaderLanguageDesc_.framebufferArrayTextures = false;
	caps_.vendor = GPUVendor::VENDOR_APPLE;
	caps_.deviceName = context_.DeviceName();
	caps_.maxTextureSize = 16384;
	caps_.coordConvention = CoordConvention::Vulkan;
	caps_.preferredDepthBufferFormat = DataFormat::D32F_S8;
	caps_.fragmentShaderFullPrecisionFloat = true;
	caps_.fragmentShaderInt32Supported = true;
	caps_.fragmentShaderDepthWriteSupported = true;
	caps_.blendMinMaxSupported = true;
	caps_.dualSourceBlend = true;
	caps_.depthClampSupported = true;
	caps_.maxClipDistances = 8;
	caps_.anisoSupported = true;
	caps_.framebufferCopySupported = true;
	caps_.framebufferDepthCopySupported = true;
	caps_.framebufferSeparateDepthCopySupported = true;
	caps_.textureDepthSupported = true;
	caps_.texture3DSupported = true;
	caps_.isTilingGPU = true;
	caps_.textureSwizzleSupported = true;
	caps_.multiSampleLevelsMask = 1;
	for (int level = 1; level <= 4; ++level) {
		if ([context_.Device() supportsTextureSampleCount:1u << level]) {
			caps_.multiSampleLevelsMask |= 1u << level;
		}
	}
	caps_.presentModesSupported = PresentMode::FIFO;
	caps_.presentMaxInterval = 1;
	if (@available(macOS 13.0, iOS 16.0, *)) {
		spatialSupported_ = [MTLFXSpatialScalerDescriptor supportsDevice:context_.Device()];
	}
	return true;
}

bool MetalDrawContext::UpscaleBoundTexture(int binding, int width, int height, UVRect *uv) {
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
		if (key != spatialKey_) {
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
				// Keep the failed key so an allocation failure does not retry every frame.
				WARN_LOG(Log::G3D, "MetalFX Spatial could not allocate %dx%d output; using normal presentation", width, height);
				spatialScaler_ = nil;
				spatialInput_ = nil;
				spatialOutput_ = nil;
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
	if (invalidation_) {
		invalidation_(InvalidationCallbackFlags::COMMAND_BUFFER_STATE);
	}
	return true;
}

void MetalDrawContext::EndPass() {
	if (encoder_) {
		[encoder_ endEncoding];
		encoder_ = nil;
	}
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
	if (!Metal::CompileShader(std::string_view((const char *)data, size), stage, {}, &compiled, &error)) {
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
	if (!desc.blend || !desc.depthStencil || !desc.raster || desc.prim > Primitive::TRIANGLE_STRIP) {
		Error("Incomplete or unsupported Metal pipeline");
		return nullptr;
	}
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	size_t requiredUniforms = 0;
	uint32_t textureMask = 0;
	for (auto module : desc.shaders) {
		auto shader = static_cast<MetalShader *>(module);
		if (shader->stage == ShaderStage::Vertex) {
			pd.vertexFunction = shader->function;
		} else if (shader->stage == ShaderStage::Fragment) {
			pd.fragmentFunction = shader->function;
		}
		for (const auto &resource : shader->compiled.resources) {
			if (resource.kind == Metal::ResourceKind::UniformBuffer) {
				requiredUniforms = std::max(requiredUniforms, (size_t)resource.byteSize);
			} else if (resource.kind == Metal::ResourceKind::SampledTexture) {
				textureMask |= 1u << resource.index;
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
	const MTLPrimitiveType types[] = {MTLPrimitiveTypePoint, MTLPrimitiveTypeLine, MTLPrimitiveTypeLineStrip, MTLPrimitiveTypeTriangle, MTLPrimitiveTypeTriangleStrip};
	pipeline->primitive = types[(size_t)desc.prim];
	return pipeline;
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
	if (!buffer || !static_cast<Metal::Buffer *>(buffer)->Update(context_.Device(), data, offset, size)) {
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
	EndPass();
	BindRef(target_, Resolve(fbo));
	pass_ = rp;
	// Execute a clear even when this pass has no draws or is immediately rebound.
	BeginPass();
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
	rp.colorAttachments[0].loadAction = actions[(size_t)pass_.color];
	const uint32_t c = pass_.clearColor;
	rp.colorAttachments[0].clearColor = MTLClearColorMake((c & 255) / 255.0, ((c >> 8) & 255) / 255.0, ((c >> 16) & 255) / 255.0, (c >> 24) / 255.0);
	if (target_->DepthStencil()) {
		rp.depthAttachment.loadAction = actions[(size_t)pass_.depth];
		rp.depthAttachment.clearDepth = pass_.clearDepth;
		rp.stencilAttachment.loadAction = actions[(size_t)pass_.stencil];
		rp.stencilAttachment.clearStencil = pass_.clearStencil;
	}
	encoder_ = [context_.Commands() renderCommandEncoderWithDescriptor:rp];
	if (!encoder_) {
		Error("Failed to begin Metal render pass");
		return false;
	}
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
	if (binding < 0 || binding >= (int)MAX_TEXTURE_SLOTS || (layer != 0 && layer != ALL_LAYERS)) {
		Error("Invalid Metal framebuffer texture binding");
		return;
	}
	auto source = Resolve(fbo);
	if (!source || (aspect != Aspect::COLOR_BIT && aspect != Aspect::DEPTH_BIT)) {
		textures_[binding] = nil;
		Error("Metal framebuffer texture binding requires color or depth");
		return;
	}
	textures_[binding] = aspect == Aspect::DEPTH_BIT ? source->DepthStencil() : source->Color();
	if (source->MultiSampleLevel() > 0) {
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
	if (sx2 - sx1 != dx2 - dx1 || sy2 - sy1 != dy2 - dy1 || sx2 <= sx1 || sy2 <= sy1) {
		return false;
	}
	return Copy(src, sx1, sy1, dst, dx1, dy1, sx2 - sx1, sy2 - sy1, aspects);
}

bool MetalDrawContext::CopyFramebufferToMemory(Framebuffer *src, Aspect aspect, int x, int y, int w, int h, DataFormat format, void *pixels, int stride, ReadbackMode mode, const char *tag) {
	EndPass();
	auto fbo = Resolve(src);
	if (!fbo) {
		return false;
	}
	std::string error;
	if (!Metal::Readback(context_, aspect == Aspect::COLOR_BIT ? fbo->Color() : fbo->DepthStencil(), aspect, x, y, w, h, format, pixels, stride, &error)) {
		Error(std::string(tag ? tag : "framebuffer readback") + ": " + error);
		return false;
	}
	return true;
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

void MetalDrawContext::BindTextures(int start, int count, Texture **textures, TextureBindFlags flags) {
	if (start < 0 || count < 0 || start > (int)MAX_TEXTURE_SLOTS - count) {
		Error("Invalid Metal texture range");
		return;
	}
	for (int i = 0; i < count; ++i) {
		textures_[start + i] = textures[i] ? static_cast<Metal::Texture *>(textures[i])->Native() : nil;
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
			*error = "Missing Metal GE texture or sampler";
			return false;
		}
		if (textures_[i] == target_->Color() || textures_[i] == target_->DepthStencil()) {
			*error = "Metal GE framebuffer feedback requires a separate copy";
			return false;
		}
	}
	for (int i = 0; i < (int)MAX_TEXTURE_SLOTS; ++i) {
		[encoder setFragmentTexture:(mask & (1u << i)) ? textures_[i] : nil atIndex:i];
		[encoder setFragmentSamplerState:(mask & (1u << i)) ? samplers_[i] : nil atIndex:i];
	}
	return true;
}

id<MTLBuffer> MetalDrawContext::Upload(const void *data, size_t size) {
	if (!data || !size) {
		return nil;
	}
	if (@available(macOS 13.0, iOS 16.0, *)) {
		if (size > context_.Device().maxBufferLength) {
			return nil;
		}
	} else {
		return nil;
	}
	return [context_.Device() newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
}

void MetalDrawContext::UpdateDynamicUniformBuffer(const void *data, size_t size) {
	uniform_ = Upload(data, size);
}

bool MetalDrawContext::Apply(id<MTLBuffer> vertices, size_t offset) {
	if (!target_ && layer_) {
		// A minimized/timed-out surface is an ordinary dropped presentation,
		// not a missing game resource or a fatal rendering error.
		return false;
	}
	if (!pipeline_ || !target_ || (pipeline_->stride && (!vertices || offset >= vertices.length)) ||
		(pipeline_->uniformSize && uniform_.length < pipeline_->uniformSize)) {
		Error("Incomplete Metal draw bindings");
		return false;
	}
	if (pipeline_->raster.cull == CullMode::FRONT_AND_BACK) {
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
		if (textures_[i] == target_->Color() || textures_[i] == target_->DepthStencil()) {
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
	id<MTLRenderPipelineState> state = pipeline_->states[key];
	if (!state) {
		pipeline_->desc.colorAttachments[0].pixelFormat = colorFormat;
		pipeline_->desc.depthAttachmentPixelFormat = depthFormat;
		pipeline_->desc.stencilAttachmentPixelFormat = depthFormat;
		pipeline_->desc.rasterSampleCount = target_->SampleCount();
		NSError *error = nil;
		state = [context_.Device() newRenderPipelineStateWithDescriptor:pipeline_->desc error:&error];
		if (!state) {
			Error(error ? error.localizedDescription.UTF8String : "Metal pipeline creation failed");
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
		pipeline_->depthStates[depthKey] = depthState;
	}
	if (!BeginPass()) {
		return false;
	}
	[encoder_ setRenderPipelineState:state];
	[encoder_ setDepthStencilState:depthState];
	[encoder_ setStencilReferenceValue:stencilRef_];
	[encoder_ setCullMode:pipeline_->raster.cull == CullMode::BACK ? MTLCullModeBack : pipeline_->raster.cull == CullMode::FRONT ? MTLCullModeFront : MTLCullModeNone];
	[encoder_ setFrontFacingWinding:pipeline_->raster.frontFace == Facing::CCW ? MTLWindingCounterClockwise : MTLWindingClockwise];
	// GE draws can enable depth clamping on this encoder; thin3d uses clipping.
	[encoder_ setDepthClipMode:MTLDepthClipModeClip];
	[encoder_ setViewport:(MTLViewport){viewport_.TopLeftX, viewport_.TopLeftY, viewport_.Width, viewport_.Height, viewport_.MinDepth, viewport_.MaxDepth}];
	[encoder_ setScissorRect:MTLScissorRect{(NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h}];
	[encoder_ setBlendColorRed:blendColor_[0] green:blendColor_[1] blue:blendColor_[2] alpha:blendColor_[3]];
	if (vertices) {
		[encoder_ setVertexBuffer:vertices offset:offset atIndex:Metal::VERTEX_BUFFER_SLOT];
	}
	if (uniform_) {
		[encoder_ setVertexBuffer:uniform_ offset:0 atIndex:0];
		[encoder_ setFragmentBuffer:uniform_ offset:0 atIndex:0];
	}
	for (int i = 0; i < (int)MAX_TEXTURE_SLOTS; ++i) {
		[encoder_ setVertexTexture:textures_[i] atIndex:i];
		[encoder_ setFragmentTexture:textures_[i] atIndex:i];
		[encoder_ setVertexSamplerState:samplers_[i] atIndex:i];
		[encoder_ setFragmentSamplerState:samplers_[i] atIndex:i];
	}
	return true;
}

void MetalDrawContext::Draw(int count, int offset) {
	if (count <= 0 || offset < 0 || vertexOffset_ < 0 || !pipeline_) {
		return;
	}
	id<MTLBuffer> vertices = vertex_ ? vertex_->Native() : nil;
	if (pipeline_->stride && (!vertices || (uint64_t)vertexOffset_ + ((uint64_t)offset + count) * pipeline_->stride > vertices.length)) {
		Error("Metal draw exceeds vertex buffer");
		return;
	}
	if (Apply(vertices, vertexOffset_)) {
		[encoder_ drawPrimitives:pipeline_->primitive vertexStart:offset vertexCount:count];
	}
}

void MetalDrawContext::DrawIndexed(int count, int offset) {
	if (count <= 0 || offset < 0 || indexOffset_ < 0 || vertexOffset_ < 0 || !index_) {
		return;
	}
	id<MTLBuffer> indices = index_->Native();
	const size_t start = (size_t)indexOffset_ + (size_t)offset * 2;
	if ((start & 1) || start > indices.length || (size_t)count > (indices.length - start) / 2) {
		Error("Metal draw exceeds index buffer");
		return;
	}
	if (Apply(vertex_ ? vertex_->Native() : nil, vertexOffset_)) {
		[encoder_ drawIndexedPrimitives:pipeline_->primitive indexCount:count indexType:MTLIndexTypeUInt16 indexBuffer:indices indexBufferOffset:start];
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
	if (Apply(vertices.buffer, vertices.offset)) {
		[encoder_ drawPrimitives:pipeline_->primitive vertexStart:0 vertexCount:count];
	}
}

void MetalDrawContext::DrawIndexedUP(const void *data, int count, const void *indices, int indexCount) {
	if (!pipeline_ || count <= 0 || indexCount <= 0 || pipeline_->stride <= 0) {
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
	auto index = context_.Upload(indices, (size_t)indexCount * 2, &error);
	if (!index) {
		Error(error);
		return;
	}
	if (Apply(vertices.buffer, vertices.offset)) {
		[encoder_ drawIndexedPrimitives:pipeline_->primitive indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:index.buffer indexBufferOffset:index.offset];
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
	Present(PresentMode::FIFO);
	attemptedDrawable_ = false;
	++frameCount_;
	Commands();
}

void MetalDrawContext::Present(PresentMode mode) {
	EndPass();
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
			if (texture == backbuffer_->Color() || texture == backbuffer_->DepthStencil()) {
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
			layer_.maximumDrawableCount = 3;
			layer_.allowsNextDrawableTimeout = YES;
#if PPSSPP_PLATFORM(MAC)
			layer_.displaySyncEnabled = YES;
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
	EndPass();
	pass_ = {(aspects & Aspect::COLOR_BIT) ? RPAction::CLEAR : RPAction::KEEP,
		(aspects & Aspect::DEPTH_BIT) ? RPAction::CLEAR : RPAction::KEEP,
		(aspects & Aspect::STENCIL_BIT) ? RPAction::CLEAR : RPAction::KEEP,
		color, depth, (uint8_t)stencil, "Metal clear"};
	BeginPass();
}

std::string MetalDrawContext::GetInfoString(InfoField info) const {
	switch (info) {
	case InfoField::APINAME: return "Metal";
	case InfoField::APIVERSION: return "3";
	case InfoField::SHADELANGVERSION: return "MSL 3.0";
	case InfoField::VENDOR: return "Apple";
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
