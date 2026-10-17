// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/metal4/ShaderBindingsMetal.h"
#include "GPU/metal4/TextureShaderMetal.h"
#include "Common/GPU/metal4/MetalBackend.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <cmath>

#include "Common/Data/Convert/ColorConv.h"
#include "Common/GPU/metal4/MetalResources.h"
#include "Common/Math/math_util.h"
#include "GPU/metal4/ShaderManagerMetal.h"
#include "GPU/metal4/TextureCacheMetal.h"
#include "Core/Config.h"
#include "Core/HDRemaster.h"
#include "Core/MemMap.h"
#include "Common/File/VFS/VFS.h"
#include "Common/StringUtils.h"
#include "GPU/metal4/PostShaderMetal.h"

bool TextureScalerMetal::Configure(Metal::RenderContext &context, const MetalTextureShaderInfo &info, std::string *error) {
	Clear();
	error->clear();
	const bool multipass = info.computeShaderFiles.size() > 1;
	const size_t expectedStages = multipass ? (info.scaleFactor == 2 ? 3 : info.scaleFactor == 4 ? 4 : 0) : 1;
	if (info.computeShaderFiles.size() != expectedStages || info.scaleFactor < 2 || info.scaleFactor >= 8) {
		*error = "Unsupported Metal texture scaling shader";
		return false;
	}
	size_t constantsSize = 0;
	if (!info.constantBuffer.empty()) {
		uint8_t *constants = g_VFS.ReadFile(info.constantBuffer.c_str(), &constantsSize);
		if (!constants || !constantsSize) {
			delete[] constants;
			*error = "Failed to read Metal texture scaling constants";
			return false;
		}
		constants_ = [context.Device() newBufferWithBytes:constants length:constantsSize options:MTLResourceStorageModeShared];
		delete[] constants;
		if (!constants_) {
			*error = "Failed to allocate Metal texture scaling constants";
			return false;
		}
	}
	for (size_t stage = 0; stage < expectedStages; ++stage) {
		size_t size = 0;
		uint8_t *data = g_VFS.ReadFile(info.computeShaderFiles[stage].c_str(), &size);
		if (!data) {
			*error = "Failed to read Metal texture scaling shader";
			Clear();
			return false;
		}
		std::string kernel((const char *)data, size);
		delete[] data;
		std::string source = multipass ? kernel : StringFromFormat(metalUploadShader, kernel.c_str());
		const std::string pushConstants = "layout(push_constant) uniform Params";
		const size_t pos = source.find(pushConstants);
		if (pos == std::string::npos) {
			*error = "Texture scaling uniform layout is missing";
			Clear();
			return false;
		}
		source.replace(pos, pushConstants.size(), "layout(std140, set=0, binding=2) uniform Params");
		Metal::ShaderCompileOptions options;
		options.textureBindingBase = 0;
		options.noInlineFunctions = info.metalNoInlineFunctions;
		if (multipass && source.find("float nnedi3(") != std::string::npos) {
			options.noInlineFunctions.push_back("nnedi3");
		}
#if PPSSPP_PLATFORM(IOS)
		options.ios = true;
#endif
		Metal::CompiledShader shader;
		if (!Metal::CompileShader(source, ShaderStage::Compute, options, &shader, error)) {
			Clear();
			return false;
		}
		// The upload wrapper uses slots 0..2. Multipass reads scratch images at 3.
		for (const auto &resource : shader.resources) {
			if (!((resource.kind == Metal::ResourceKind::StorageTexture && resource.index == 0) ||
				(resource.kind == Metal::ResourceKind::StorageBuffer && resource.index == 1 && stage == 0) ||
				(resource.kind == Metal::ResourceKind::StorageTexture && resource.index == 3 && multipass && stage > 0) ||
				(resource.kind == Metal::ResourceKind::UniformBuffer && resource.index == 2 && resource.byteSize <= 16) ||
				(resource.kind == Metal::ResourceKind::UniformBuffer && resource.index == 4 && constants_ && resource.byteSize <= constantsSize))) {
				*error = "Unsupported Metal texture scaling resource binding";
				Clear();
				return false;
			}
		}
		if (shader.workgroupSize[0] != 8 || shader.workgroupSize[1] != 8 || shader.workgroupSize[2] != 1) {
			*error = "Unsupported Metal texture scaling workgroup size";
			Clear();
			return false;
		}
		auto function = context.CreateShader(shader, info.name.c_str(), error);
		if (!function) {
			Clear();
			return false;
		}
		NSError *nativeError = nil;
		pipelines_[stage] = context.CreateComputePipeline(function, &nativeError);
		if (!pipelines_[stage] || pipelines_[stage].maxTotalThreadsPerThreadgroup < 64) {
			*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to create Metal texture scaling pipeline";
			Clear();
			return false;
		}
	}
	pipelineCount_ = (int)expectedStages;
	scaleFactor_ = info.scaleFactor;
	return true;
}

Metal::Texture *TextureScalerMetal::Scale(Metal::RenderContext &context, Metal::UploadSlice input, int width, int height, int mipLevels, std::string *error) {
	error->clear();
	const int maxDimension = Metal::MaxTextureDimension(context.Device());
	if (!pipelineCount_ || !input || width <= 0 || height <= 0 || width > maxDimension / scaleFactor_ || height > maxDimension / scaleFactor_) {
		*error = "Invalid Metal texture scaling input";
		return nullptr;
	}
	if (!context.Commands()) {
		*error = "Metal texture scaling requires active commands";
		return nullptr;
	}
	Draw::TextureDesc desc{};
	desc.type = Draw::TextureType::LINEAR2D;
	desc.width = width * scaleFactor_;
	desc.height = height * scaleFactor_;
	desc.depth = 1;
	desc.mipLevels = mipLevels;
	desc.format = Draw::DataFormat::R8G8B8A8_UNORM;
	desc.tag = "Upscaled PSP texture";
	auto texture = Metal::Texture::Create(context, desc, error, true);
	if (!texture) {
		return nullptr;
	}
	// The output is a new allocation. Initialize it before the active rendering
	// buffer, just like ordinary texture uploads, without breaking its render pass.
	auto commands = context.InitializationCommands(error);
	if (!commands) {
		texture->Release();
		return nullptr;
	}
	if (pipelineCount_ == 1) {
		auto encoder = context.ComputeEncoder(commands, "Texture scaling");
		if (!encoder) {
			*error = "Failed to create Metal texture scaling encoder";
			texture->Release();
			return nullptr;
		}
		const int params[4] = { width, height, 0, 0 };
		context.Keep(pipelines_[0]);
		[encoder setComputePipelineState:pipelines_[0]];
		context.SetBuffer(encoder, input.buffer, input.offset, 1);
		context.SetBytes(encoder, params, sizeof(params), 2);
		if (constants_) {
			context.SetBuffer(encoder, constants_, 0, 4);
		}
		context.SetTexture(encoder, texture->Native(), 0);
		[encoder dispatchThreadgroups:MTLSizeMake((width + 7) / 8, (height + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
		context.EndEncoder(encoder);
	} else {
		struct ScratchDesc { int widthScale, heightScale; };
		struct StageDesc {
			int shaderIndex, inputScratch, outputScratch;
			int srcWidthScale, srcHeightScale, dstWidthScale, dstHeightScale;
			bool useFinalOutputSize;
		};
		static constexpr ScratchDesc scratch2x[] = { {1, 2}, {2, 2} };
		static constexpr ScratchDesc scratch4x[] = { {1, 2}, {2, 2}, {2, 4}, {4, 4} };
		static constexpr StageDesc stages2x[] = {
			{0, -1, 0, 1, 1, 1, 2, false},
			{1, 0, 1, 1, 2, 2, 2, false},
			{2, 1, -1, 2, 2, 2, 2, false},
		};
		static constexpr StageDesc stages4x[] = {
			{0, -1, 0, 1, 1, 1, 2, false},
			{2, 0, 1, 1, 2, 2, 2, false},
			{1, 1, 2, 2, 2, 2, 4, false},
			{2, 2, 3, 2, 4, 4, 4, false},
			{3, 3, -1, 4, 4, 4, 4, true},
		};
		const ScratchDesc *scratchDescs = scaleFactor_ == 2 ? scratch2x : scratch4x;
		const StageDesc *stages = scaleFactor_ == 2 ? stages2x : stages4x;
		const int scratchCount = scaleFactor_ == 2 ? 2 : 4;
		const int stageCount = scaleFactor_ == 2 ? 3 : 5;
		std::array<Draw::AutoRef<Metal::Texture>, 4> scratch;
		for (int i = 0; i < scratchCount; ++i) {
			Draw::TextureDesc scratchDesc{};
			scratchDesc.type = Draw::TextureType::LINEAR2D;
			scratchDesc.width = width * scratchDescs[i].widthScale;
			scratchDesc.height = height * scratchDescs[i].heightScale;
			scratchDesc.depth = 1;
			scratchDesc.mipLevels = 1;
			scratchDesc.format = Draw::DataFormat::R16G16B16A16_FLOAT;
			scratchDesc.tag = "Metal texture scaling scratch";
			scratch[i].reset(Metal::Texture::Create(context, scratchDesc, error, true));
			if (!scratch[i]) {
				texture->Release();
				return nullptr;
			}
		}
		// A serial compute encoder orders these dispatches and makes each
		// scratch texture available to the following stage.
		auto pass = context.ComputeEncoder(commands, "Multistage texture scaling");
		if (!pass) {
			*error = "Failed to create Metal texture scaling pass";
			texture->Release();
			return nullptr;
		}
		for (int i = 0; i < stageCount; ++i) {
			const StageDesc &stage = stages[i];
			id<MTLTexture> output = stage.outputScratch < 0 ? texture->Native() : scratch[stage.outputScratch]->Native();
			id<MTLTexture> source = stage.inputScratch < 0 ? nil : scratch[stage.inputScratch]->Native();
			const int params[4] = {
				width * stage.srcWidthScale,
				height * stage.srcHeightScale,
				stage.useFinalOutputSize ? desc.width : width * stage.dstWidthScale,
				stage.useFinalOutputSize ? desc.height : height * stage.dstHeightScale,
			};
			context.Keep(pipelines_[stage.shaderIndex]);
			[pass setComputePipelineState:pipelines_[stage.shaderIndex]];
			context.SetTexture(pass, output, 0);
			if (stage.inputScratch < 0) {
				context.SetBuffer(pass, input.buffer, input.offset, 1);
			} else {
				context.SetTexture(pass, source, 3);
			}
			context.SetBytes(pass, params, sizeof(params), 2);
			if (constants_) {
				context.SetBuffer(pass, constants_, 0, 4);
			}
			[pass dispatchThreadgroups:MTLSizeMake((params[2] + 7) / 8, (params[3] + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
		}
		context.EndEncoder(pass);
	}
	if (mipLevels > 1) {
		auto blit = context.BlitEncoder(commands, "Scaled texture mipmaps");
		if (!blit) {
			*error = "Failed to encode Metal texture scaling mipmaps";
			texture->Release();
			return nullptr;
		}
		context.Keep(texture->Native());
		[blit generateMipmapsForTexture:texture->Native()];
		context.EndEncoder(blit);
	}
	return texture;
}

void TextureCacheMetal::UpdateScalingShader() {
	const std::string name = g_Config.bTexHardwareScaling ? g_Config.sTextureShaderName : "";
	if (name == scalingShader_ && (name.empty() || name == "Off" || shaderScaleFactor_ > 1)) {
		return;
	}
	scalingShader_ = name;
#if PPSSPP_PLATFORM(MAC)
	ClearNextFrame();
#endif
	textureScaler_.Clear();
	shaderScaleFactor_ = 0;
	if (!manager_ || name.empty() || name == "Off") {
		return;
	}
	MetalReloadAllPostShaderInfo(draw_);
	const MetalTextureShaderInfo *info = MetalGetTextureShaderInfo(name);
	std::string error;
	if (!info || !textureScaler_.Configure(manager_->Context(), *info, &error)) {
		WARN_LOG(Log::G3D, "Metal texture upscaler '%s' unavailable: %s", name.c_str(), error.c_str());
		return;
	}
	shaderScaleFactor_ = textureScaler_.ScaleFactor();
	INFO_LOG(Log::G3D, "Metal texture upscaler ready: %s (%dx)", name.c_str(), shaderScaleFactor_);
}

id<MTLSamplerState> SamplerCacheMetal::GetOrCreate(id<MTLDevice> device, const SamplerCacheKey &key, std::string *error) {
	error->clear();
	if (!device) {
		*error = "Metal sampler requires a device";
		return nil;
	}
	// Bias is a shader uniform on Metal, not part of MTLSamplerDescriptor.
	SamplerCacheKey nativeKey = key;
	nativeKey.lodBias = 0;
	const int aniso = key.aniso ? 1 << std::clamp((int)key.anisoLevel, 0, 4) : 1;
	// Cache the effective anisotropy separately, including the device limit.
	nativeKey.aniso = false;
	nativeKey.anisoLevel = 0;
	const bool useMips = key.mipEnable || key.maxLevel == 9 * 256;
	if (!useMips) {
		// These fields do not affect a non-mipmapped Metal sampler.
		nativeKey.minLevel = 0;
		nativeKey.maxLevel = 0;
		nativeKey.mipFilt = false;
	}
	const auto cacheKey = std::make_pair(nativeKey.fullKey, aniso);
	auto found = cache_.find(cacheKey);
	if (found != cache_.end()) {
		return found->second;
	}
	MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
	desc.sAddressMode = key.sClamp ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
	desc.tAddressMode = key.tClamp ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
	desc.rAddressMode = key.texture3d ? MTLSamplerAddressModeClampToEdge : desc.sAddressMode;
	desc.minFilter = key.minFilt ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	desc.magFilter = key.magFilt ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	// Auto Max Quality requests the full mip chain even when the PSP game did
	// not enable mipmaps. The shared key marks this by setting maxLevel to 9.
	desc.mipFilter = !useMips ? MTLSamplerMipFilterNotMipmapped : key.mipFilt ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
	desc.lodMinClamp = useMips ? std::max(0.0f, key.minLevel / 256.0f) : 0.0f;
	desc.lodMaxClamp = !useMips ? 0.0f : key.maxLevel == 9 * 256 ? FLT_MAX : std::max(desc.lodMinClamp, key.maxLevel / 256.0f);
	desc.maxAnisotropy = aniso;
	desc.supportArgumentBuffers = YES;
	auto sampler = [device newSamplerStateWithDescriptor:desc];
	if (!sampler) {
		*error = "Failed to create Metal PSP sampler";
		return nil;
	}
	cache_.emplace(cacheKey, sampler);
	return sampler;
}

TextureCacheMetal::TextureCacheMetal(Draw::DrawContext *draw, Draw2D *draw2D)
	: MetalTextureCacheCommon(draw, draw2D) {
	manager_ = (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
	framebufferManager_ = nullptr;
	shaderManager_ = nullptr;
	standardScaleFactor_ = 1;
}

TextureCacheMetal::~TextureCacheMetal() {
	Clear(true);
	ForgetLastTexture();
}

void TextureCacheMetal::DeviceLost() {
	MetalTextureCacheCommon::DeviceLost();
	ForgetLastTexture();
	ResetGETextureBindings();
	missingTexture_ = nil;
	missingTexture3D_ = nil;
	samplers_.Clear();
	textureScaler_.Clear();
	debugReadbackPipeline_ = nil;
	scalingShader_.clear();
	shaderScaleFactor_ = 0;
	manager_ = nullptr;
	draw_ = nullptr;
}

void TextureCacheMetal::DeviceRestore(Draw::DrawContext *draw) {
	MetalTextureCacheCommon::DeviceRestore(draw);
	ResetGETextureBindings();
	manager_ = (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
	missingTexture_ = nil;
	missingTexture3D_ = nil;
	samplers_.Clear();
	debugReadbackPipeline_ = nil;
	ForgetLastTexture();
	UpdateScalingShader();
}

void TextureCacheMetal::NotifyConfigChanged() {
	MetalTextureCacheCommon::NotifyConfigChanged();
	ResetGETextureBindings();
	samplers_.Clear();
	UpdateScalingShader();
}

void TextureCacheMetal::ForgetLastTexture() {
	ResetGETextureBindings();
	gstate_c.Dirty(DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS);
	if (draw_ && manager_) {
		for (int i = 0; i < 3; ++i) {
			draw_->BindNativeTexture(i, nullptr);
			manager_->SetNativeSampler(i, nil);
		}
	}
}

void TextureCacheMetal::ResetGETextureBindings() {
	temporaryRegion_.reset(nullptr);
	geScaledTexture_ = nullptr;
	geTexture_ = nil;
	geSampler_ = nil;
	geClutTexture_ = nil;
	geClutSampler_ = nil;
}

void TextureCacheMetal::RestoreGETextureBindings() {
	// The shared depal path invalidates thin3d after binding the GE texture.
	// Rebind its native resources after the last thin3d operation and before the GE draw.
	draw_->BindNativeTexture(ShaderBinding::DRAW_BINDING_TEXTURE, (__bridge void *)geTexture_);
	manager_->SetNativeSampler(ShaderBinding::DRAW_BINDING_TEXTURE, geSampler_);
	draw_->BindNativeTexture(ShaderBinding::DRAW_BINDING_DEPAL_TEXTURE, (__bridge void *)geClutTexture_);
	manager_->SetNativeSampler(ShaderBinding::DRAW_BINDING_DEPAL_TEXTURE, geClutSampler_);
}

bool TextureCacheMetal::BindSamplingRegion(const TransformedVertex *vertices, TransformedVertex *adjusted, const MTLViewport &viewport) {
#if PPSSPP_PLATFORM(MAC)
	temporaryRegion_.reset(nullptr);
	if (!geScaledTexture_) {
		return false;
	}
	// A previous quad may have bound another region of this same PSP texture.
	geTexture_ = geScaledTexture_->Native();
	if (!vertices || geSamplerKey_.minFilt || geSamplerKey_.magFilt || geSamplerKey_.mipEnable ||
		gstate_c.textureIsFramebuffer || gstate_c.textureIsArray || gstate_c.needShaderTexClamp || gstate_c.curTextureIs3D ||
		viewport.width <= 0 || viewport.height <= 0 || gstate_c.curRTWidth <= 0 || gstate_c.curRTHeight <= 0) {
		return false;
	}
	auto &source = scaledSources_.at(geScaledTexture_);
	float minU = vertices[0].u, maxU = minU;
	float minV = vertices[0].v, maxV = minV;
	for (int i = 0; i < 4; ++i) {
		const auto &v = vertices[i];
		if (v.pos_w != 1.0f || v.uv_w != 1.0f || !std::isfinite(v.u) || !std::isfinite(v.v) || !std::isfinite(v.x) || !std::isfinite(v.y)) {
			return false;
		}
		minU = std::min(minU, v.u);
		maxU = std::max(maxU, v.u);
		minV = std::min(minV, v.v);
		maxV = std::max(maxV, v.v);
	}
	if (minU < 0.0f || minV < 0.0f || maxU > 1.0f || maxV > 1.0f || minU == maxU || minV == maxV) {
		return false;
	}
	unsigned corners = 0;
	for (int i = 0; i < 4; ++i) {
		const auto &v = vertices[i];
		if ((v.u != minU && v.u != maxU) || (v.v != minV && v.v != maxV)) {
			return false;
		}
		corners |= 1u << ((v.u == maxU ? 1 : 0) | (v.v == maxV ? 2 : 0));
	}
	if (corners != 15) {
		return false;
	}
	float minX = vertices[0].x, maxX = minX, minY = vertices[0].y, maxY = minY;
	for (int i = 1; i < 4; ++i) {
		minX = std::min(minX, vertices[i].x); maxX = std::max(maxX, vertices[i].x);
		minY = std::min(minY, vertices[i].y); maxY = std::max(maxY, vertices[i].y);
	}
	if (minX == maxX || minY == maxY) {
		return false;
	}
	const TransformedVertex *xy[4]{};
	for (int i = 0; i < 4; ++i) {
		if ((vertices[i].x != minX && vertices[i].x != maxX) || (vertices[i].y != minY && vertices[i].y != maxY)) {
			return false;
		}
		xy[(vertices[i].x == maxX ? 1 : 0) | (vertices[i].y == maxY ? 2 : 0)] = &vertices[i];
	}
	if (!xy[0] || !xy[1] || !xy[2] || !xy[3] ||
		xy[0]->u + xy[3]->u != xy[1]->u + xy[2]->u || xy[0]->v + xy[3]->v != xy[1]->v + xy[2]->v) {
		return false;
	}
	const double offsetX = gstate.isModeThrough() ? 0 : gstate.getOffsetX();
	const double offsetY = gstate.isModeThrough() ? 0 : gstate.getOffsetY();
	const double leftX = (minX - offsetX + gstate_c.curRTOffsetX) * viewport.width / gstate_c.curRTWidth + viewport.originX;
	const double rightX = (maxX - offsetX + gstate_c.curRTOffsetX) * viewport.width / gstate_c.curRTWidth + viewport.originX;
	const double topY = (minY - offsetY + gstate_c.curRTOffsetY) * viewport.height / gstate_c.curRTHeight + viewport.originY;
	const double bottomY = (maxY - offsetY + gstate_c.curRTOffsetY) * viewport.height / gstate_c.curRTHeight + viewport.originY;
	MTLSamplePosition samples[9]{{0.5f, 0.5f}};
	const int sampleCount = manager_->SampleCount();
	if (sampleCount < 1 || sampleCount > 8) {
		return false;
	}
	if (sampleCount > 1) {
		[manager_->Context().Device() getDefaultSamplePositions:samples count:sampleCount];
		// Centroid interpolation can also use the pixel center when it is inside.
		samples[sampleCount] = {0.5f, 0.5f};
	}
	double firstX = rightX, lastX = leftX, firstY = bottomY, lastY = topY;
	for (int i = 0; i < (sampleCount > 1 ? sampleCount + 1 : 1); ++i) {
		const double x1 = std::ceil(leftX - samples[i].x) + samples[i].x;
		const double x2 = std::ceil(rightX - samples[i].x) - 1.0 + samples[i].x;
		const double y1 = std::ceil(topY - samples[i].y) + samples[i].y;
		const double y2 = std::ceil(bottomY - samples[i].y) - 1.0 + samples[i].y;
		if (x1 <= x2 && y1 <= y2) {
			firstX = std::min(firstX, x1);
			lastX = std::max(lastX, x2);
			firstY = std::min(firstY, y1);
			lastY = std::max(lastY, y2);
		}
	}
	if (firstX > lastX || firstY > lastY) {
		return false;
	}
	// Only point-sampled texels belong to this sprite. Bound that footprint
	// before scaling, so a texture atlas's transparent gutters stay outside it.
	double sampleMinU = maxU, sampleMaxU = minU, sampleMinV = maxV, sampleMaxV = minV;
	for (double sx : {firstX, lastX}) {
		for (double sy : {firstY, lastY}) {
			const double tx = (sx - leftX) / (rightX - leftX), ty = (sy - topY) / (bottomY - topY);
			const double u = xy[0]->u + (xy[1]->u - xy[0]->u) * tx + (xy[2]->u - xy[0]->u) * ty;
			const double v = xy[0]->v + (xy[1]->v - xy[0]->v) * tx + (xy[2]->v - xy[0]->v) * ty;
			sampleMinU = std::min(sampleMinU, u); sampleMaxU = std::max(sampleMaxU, u);
			sampleMinV = std::min(sampleMinV, v); sampleMaxV = std::max(sampleMaxV, v);
		}
	}
	const int x = std::clamp((int)std::floor(sampleMinU * source.width), 0, source.width - 1);
	const int y = std::clamp((int)std::floor(sampleMinV * source.height), 0, source.height - 1);
	const int right = std::clamp((int)std::floor(sampleMaxU * source.width) + 1, x + 1, source.width);
	const int bottom = std::clamp((int)std::floor(sampleMaxV * source.height) + 1, y + 1, source.height);
	const int width = right - x, height = bottom - y;
	if (width <= 0 || height <= 0 || (width == source.width && height == source.height)) {
		return false;
	}
	const std::array<int, 4> key{x, y, right, bottom};
	auto region = source.regions.find(key);
	id<MTLTexture> uncached = nil;
	if (region == source.regions.end()) {
		std::string error;
		auto &context = manager_->Context();
		auto input = context.ReserveUpload((size_t)width * height * 4, &error);
		if (!input) {
			return false;
		}
		auto *dest = (uint32_t *)((uint8_t *)input.buffer.contents + input.offset);
		for (int row = 0; row < height; ++row) {
			memcpy(dest + row * width, source.pixels.data() + (y + row) * source.width + x, width * 4);
		}
		Draw::AutoRef<Metal::Texture> texture;
		texture.reset(textureScaler_.Scale(context, input, width, height, 1, &error));
		if (!texture) {
			WARN_LOG(Log::G3D, "Metal atlas region scaling failed: %s", error.c_str());
			return false;
		}
		const size_t bytes = texture->Native().width * texture->Native().height * 4;
		constexpr size_t REGION_CACHE_BYTES = 16 * 1024 * 1024;
		// Encoded draws retain their textures until completion. Dropping cached
		// regions here bounds retained memory without synchronizing with the GPU.
		if (regionBytes_ + bytes > REGION_CACHE_BYTES || regionCount_ >= 256) {
			for (auto &item : scaledSources_) {
				item.second.regions.clear();
			}
			regionBytes_ = 0;
			regionCount_ = 0;
		}
		if (bytes > REGION_CACHE_BYTES) {
			temporaryRegion_ = texture;
			uncached = texture->Native();
		} else {
			region = source.regions.emplace(key, texture).first;
			regionBytes_ += bytes;
			++regionCount_;
		}
	}
	geTexture_ = uncached ? uncached : region->second->Native();
	for (int i = 0; i < 4; ++i) {
		adjusted[i] = vertices[i];
		adjusted[i].u = (vertices[i].u * source.width - x) / width;
		adjusted[i].v = (vertices[i].v * source.height - y) / height;
	}
	return true;
#else
	return false;
#endif
}

void TextureCacheMetal::BoundFramebufferTexture() {
	geScaledTexture_ = nullptr;
	geTexture_ = (__bridge id<MTLTexture>)(void *)draw_->GetNativeObject(Draw::NativeObject::BOUND_TEXTURE0_IMAGEVIEW);
}

const size_t TextureCacheMetal::CacheSizeEstimate() const {
	const size_t temporaryBytes = temporaryRegion_ ? temporaryRegion_->Native().width * temporaryRegion_->Native().height * 4 : 0;
	return MetalTextureCacheCommon::CacheSizeEstimate() + sourceBytes_ + regionBytes_ + temporaryBytes;
}

void TextureCacheMetal::BindTexture(MetalTexCacheEntry *entry) {
	geScaledTexture_ = nullptr;
	if (!draw_ || !manager_) {
		return;
	}
	if (entry && entry->texturePtr) {
		auto *texture = static_cast<Metal::Texture *>(entry->texturePtr);
		geScaledTexture_ = scaledSources_.find(texture) != scaledSources_.end() ? texture : nullptr;
		geTexture_ = (__bridge id<MTLTexture>)GetNativeTextureView(entry, false);
		draw_->BindNativeTexture(ShaderBinding::DRAW_BINDING_TEXTURE, (__bridge void *)geTexture_);
		return;
	}
	const bool is3D = entry && (entry->status & MetalTexStatus::IS_3D) != 0;
	id<MTLTexture> fallback = is3D ? missingTexture3D_ : missingTexture_;
	if (!fallback) {
		MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
			width:1 height:1 mipmapped:NO];
		if (is3D) {
			desc.textureType = MTLTextureType3D;
			desc.depth = 1;
		}
		desc.storageMode = MTLStorageModeShared;
		desc.usage = MTLTextureUsageShaderRead;
		fallback = [manager_->Context().Device() newTextureWithDescriptor:desc];
		if (fallback) {
			const uint32_t transparentBlack = 0;
			if (is3D) {
				[fallback replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:0
					withBytes:&transparentBlack bytesPerRow:sizeof(transparentBlack) bytesPerImage:sizeof(transparentBlack)];
			} else {
				[fallback replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0
					withBytes:&transparentBlack bytesPerRow:sizeof(transparentBlack)];
			}
			if (is3D) {
				missingTexture3D_ = fallback;
			} else {
				missingTexture_ = fallback;
			}
		} else {
			ERROR_LOG(Log::G3D, "Failed to create Metal missing-texture fallback");
		}
	}
	draw_->BindNativeTexture(ShaderBinding::DRAW_BINDING_TEXTURE, (__bridge void *)fallback);
	geTexture_ = fallback;
}

void TextureCacheMetal::Unbind() {
	if (draw_ && manager_) {
		// GE shaders still sample when the PSP texture address is invalid.
		BindTexture(nullptr);
		manager_->SetNativeSampler(ShaderBinding::DRAW_BINDING_TEXTURE, nil);
		geSampler_ = nil;
	}
}

void TextureCacheMetal::ReleaseTexture(MetalTexCacheEntry *entry, bool deleteThem) {
	if (entry->texturePtr) {
		auto *texture = static_cast<Metal::Texture *>(entry->texturePtr);
		auto source = scaledSources_.find(texture);
		if (source != scaledSources_.end()) {
			sourceBytes_ -= source->second.pixels.size() * sizeof(uint32_t);
			regionCount_ -= source->second.regions.size();
			for (const auto &region : source->second.regions) {
				regionBytes_ -= region.second->Native().width * region.second->Native().height * 4;
			}
			scaledSources_.erase(source);
		}
		if (geScaledTexture_ == texture) {
			geScaledTexture_ = nullptr;
		}
		static_cast<Metal::Texture *>(entry->texturePtr)->Release();
		entry->texturePtr = nullptr;
	}
}

void *TextureCacheMetal::GetNativeTextureView(const MetalTexCacheEntry *entry, bool flat) const {
	if (!entry || !entry->texturePtr) {
		return nullptr;
	}
	auto texture = static_cast<Metal::Texture *>(entry->texturePtr);
	return (__bridge void *)(!flat && gstate_c.textureIsArray && gstate_c.Use(GPU_USE_FRAMEBUFFER_ARRAYS) ?
		texture->ArrayView() : texture->Native());
}

void TextureCacheMetal::ApplySamplerByKey(const SamplerCacheKey &key) {
	geSamplerKey_ = key;
	if (!manager_) {
		return;
	}
	std::string error;
	auto sampler = samplers_.GetOrCreate(manager_->Context().Device(), key, &error);
	if (!sampler) {
		ERROR_LOG(Log::G3D, "%s", error.c_str());
	}
	manager_->SetNativeSampler(ShaderBinding::DRAW_BINDING_TEXTURE, sampler);
	geSampler_ = sampler;
	if (shaderManager_) {
		static_cast<ShaderManagerMetal *>(shaderManager_)->SetSamplerLodBias(key.mipEnable ? key.lodBias / 256.0f : 0.0f);
	}
}

void TextureCacheMetal::BindAsClutTexture(Draw::Texture *texture, bool smooth) {
	if (!manager_) {
		return;
	}
	draw_->BindTexture(ShaderBinding::DRAW_BINDING_DEPAL_TEXTURE, texture);
	geClutTexture_ = texture ? static_cast<Metal::Texture *>(texture)->Native() : nil;
	SamplerCacheKey key{};
	key.sClamp = true;
	key.tClamp = true;
	key.minFilt = smooth;
	key.magFilt = smooth;
	std::string error;
	geClutSampler_ = samplers_.GetOrCreate(manager_->Context().Device(), key, &error);
	manager_->SetNativeSampler(ShaderBinding::DRAW_BINDING_DEPAL_TEXTURE, geClutSampler_);
	if (!error.empty()) {
		ERROR_LOG(Log::G3D, "%s", error.c_str());
	}
}

void TextureCacheMetal::BuildTexture(MetalTexCacheEntry *entry) {
	MetalBuildTexturePlan plan;
	plan.hardwareScaling = g_Config.bTexHardwareScaling && textureScaler_.ScaleFactor() > 1;
	plan.slowScaler = !plan.hardwareScaling;
	if (!manager_ || !PrepareBuildTexture(plan, entry)) {
		return;
	}
	const bool maxQualityMips = !plan.decodeToClut8 && g_Config.iTexFiltering == TEX_FILTER_AUTO_MAX_QUALITY && plan.w <= 256 && plan.h <= 256 &&
		(!plan.doReplace || plan.replaced->Format() == Draw::DataFormat::R8G8B8A8_UNORM);
	const bool generateMaxQualityMips = maxQualityMips && plan.maxPossibleLevels > plan.levelsToCreate;
	if (maxQualityMips) {
		plan.levelsToCreate = plan.maxPossibleLevels;
	}
	if (plan.hardwareScaling && plan.scaleFactor > 1) {
		// Decode through the common path so CLUT expansion, swizzling, alpha
		// tracking and texture dumping retain their existing semantics.
		MetalBuildTexturePlan decodePlan = plan;
		decodePlan.scaleFactor = 1;
		const int width = (entry->status & MetalTexStatus::PSP_SIZE_CLIPPED) ? std::min(gstate.getTextureWidth(plan.baseLevelSrc), 512) : gstate.getTextureWidth(plan.baseLevelSrc);
		const int height = (entry->status & MetalTexStatus::PSP_SIZE_CLIPPED) ? std::min(gstate.getTextureHeight(plan.baseLevelSrc), 512) : gstate.getTextureHeight(plan.baseLevelSrc);
		std::string error;
		const int levels = std::min(plan.levelsToCreate, plan.maxPossibleLevels);
		ReleaseTexture(entry, true);
		auto &context = manager_->Context();
		Metal::Texture *texture = nullptr;
		if (width > 0 && height > 0 && (context.Commands() || context.BeginCommands(&error))) {
			const size_t uploadBytes = (size_t)width * height * sizeof(uint32_t);
			auto input = context.ReserveUpload(uploadBytes, &error);
			if (input) {
				uint8_t *pixels = (uint8_t *)input.buffer.contents + input.offset;
				LoadTextureLevel(*entry, pixels, uploadBytes, width * 4,
					decodePlan, plan.baseLevelSrc, Draw::DataFormat::R8G8B8A8_UNORM, TexDecodeFlags{});
				texture = textureScaler_.Scale(context, input, width, height, levels, &error);
#if PPSSPP_PLATFORM(MAC)
				if (texture) {
					auto &source = scaledSources_[texture];
					source.width = width;
					source.height = height;
					source.pixels.resize((size_t)width * height);
					memcpy(source.pixels.data(), pixels, uploadBytes);
					sourceBytes_ += uploadBytes;
				}
#endif
			}
		} else if (error.empty()) {
			error = "Invalid Metal texture scaling size";
		}
		if (texture) {
			entry->texturePtr = texture;
			entry->status &= ~MetalTexStatus::IS_3D;
			if (levels == 1) {
				entry->status |= MetalTexStatus::NO_MIPS;
			} else {
				entry->status &= ~MetalTexStatus::NO_MIPS;
			}
			return;
		}
		WARN_LOG(Log::G3D, "Metal texture scaling failed: %s", error.c_str());
		plan.createW /= plan.scaleFactor;
		plan.createH /= plan.scaleFactor;
		plan.scaleFactor = 1;
		plan.maxPossibleLevels = log2i(std::max(plan.createW, plan.createH)) + 1;
		plan.levelsToCreate = std::min(plan.levelsToCreate, plan.maxPossibleLevels);
		entry->status &= ~MetalTexStatus::IS_SCALED_OR_REPLACED;
	}
	const int maxDimension = Metal::MaxTextureDimension(manager_->Context().Device());
	if (plan.depth <= 0 || plan.levelsToLoad <= 0 || plan.createW <= 0 || plan.createH <= 0 ||
		plan.createW > maxDimension || plan.createH > maxDimension) {
		ERROR_LOG(Log::G3D, "Unsupported Metal PSP texture dimensions or mip chain");
		return;
	}
	Draw::TextureDesc desc{};
	desc.type = plan.depth > 1 ? Draw::TextureType::LINEAR3D : Draw::TextureType::LINEAR2D;
	desc.width = plan.createW;
	desc.height = plan.createH;
	desc.depth = plan.depth;
	desc.format = plan.decodeToClut8 ? Draw::DataFormat::R8_UNORM : Draw::DataFormat::R8G8B8A8_UNORM;
	if (gstate_c.Use(GPU_USE_16BIT_FORMATS) && !plan.decodeToClut8 && !plan.doReplace && !plan.saveTexture &&
		plan.scaleFactor == 1 && !generateMaxQualityMips) {
		int format = entry->format;
		if (format >= GE_TFMT_CLUT4 && format <= GE_TFMT_CLUT32) {
			format = gstate.getClutPaletteFormat();
		}
		switch (format) {
		case GE_TFMT_5650: desc.format = Draw::DataFormat::R5G6B5_UNORM_PACK16; break;
		case GE_TFMT_5551: desc.format = Draw::DataFormat::R5G5B5A1_UNORM_PACK16; break;
		case GE_TFMT_4444: desc.format = Draw::DataFormat::R4G4B4A4_UNORM_PACK16; break;
		default: break;
		}
	}
	if (plan.doReplace && !plan.decodeToClut8) {
		desc.format = plan.replaced->Format();
		if (!(draw_->GetDataFormatSupport(desc.format) & Draw::FMT_TEXTURE)) {
			ERROR_LOG(Log::G3D, "Unsupported replacement texture format for Metal");
			return;
		}
	}
	int replacementBlockSize = 0;
	bool compressedReplacement = Metal::IsBlockCompressed(desc.format, &replacementBlockSize);
	desc.mipLevels = std::min(plan.levelsToCreate, plan.maxPossibleLevels);
	desc.generateMips = !compressedReplacement && desc.mipLevels > plan.levelsToLoad &&
		(draw_->GetDataFormatSupport(desc.format) & Draw::FMT_AUTOGEN_MIPS);
	if (!desc.generateMips) {
		desc.mipLevels = std::min(desc.mipLevels, plan.levelsToLoad);
	}
	if (desc.mipLevels <= 0) {
		return;
	}
	desc.tag = "PSP texture";
	desc.initData.resize(std::min(plan.levelsToLoad, desc.mipLevels), nullptr);
	// The common decoder preserves PSP bit order, including paletted colors.
	// Reuse GL's packed conversions without quantizing the channels to RGBA8.
	auto convertPacked = [&](uint8_t *data, uint32_t w, uint32_t h, uint32_t pitch) {
		for (uint32_t y = 0; y < h; ++y) {
			auto row = (uint16_t *)(data + y * pitch);
			switch (desc.format) {
			case Draw::DataFormat::R5G6B5_UNORM_PACK16: ConvertRGB565ToBGR565(row, row, w); break;
			case Draw::DataFormat::R5G5B5A1_UNORM_PACK16: ConvertRGBA5551ToABGR1555(row, row, w); break;
			case Draw::DataFormat::R4G4B4A4_UNORM_PACK16: ConvertRGBA4444ToABGR4444(row, row, w); break;
			default: return;
			}
		}
	};
	int level = 0;
	desc.initDataCallback = [&](uint8_t *data, const uint8_t *, uint32_t w, uint32_t h, uint32_t d, uint32_t pitch, uint32_t slicePitch) {
		if (plan.depth > 1) {
			if (level != 0 || w != plan.createW || h != plan.createH || d != plan.depth) {
				return false;
			}
			// Equal-size PSP mip levels are the slices of one volume, as on GL/Vulkan.
			for (uint32_t z = 0; z < d; ++z) {
				LoadTextureLevel(*entry, data + slicePitch * z, slicePitch, pitch, plan, z, desc.format, TexDecodeFlags{});
				if (!plan.doReplace) {
					convertPacked(data + slicePitch * z, w, h, pitch);
				}
			}
			++level;
			return true;
		}
		const int srcLevel = level == 0 ? plan.baseLevelSrc : level;
		int expectedW, expectedH;
		plan.GetMipSize(level, &expectedW, &expectedH);
		if (w != std::max(1, expectedW) || h != std::max(1, expectedH) || d != 1) {
			return false;
		}
		if (compressedReplacement) {
			LoadTextureLevel(*entry, data, slicePitch, (int)pitch, plan, srcLevel, desc.format, TexDecodeFlags{}, true);
			++level;
			return true;
		}
		LoadTextureLevel(*entry, data, slicePitch, pitch, plan, srcLevel, desc.format, TexDecodeFlags{});
		if (!plan.doReplace) {
			convertPacked(data, w, h, pitch);
		}
		++level;
		return true;
	};
	// thin3d initializes this new allocation before the active render buffer.
	ReleaseTexture(entry, true);
	std::string error;
	auto texture = Metal::Texture::Create(manager_->Context(), desc, &error);
	if (!texture && error == "Failed to allocate Metal texture") {
		WARN_LOG(Log::G3D, "Metal texture allocation failed; decimating and trying the PSP texture at 1x");
		// Decimation may purge replacement data. The retry must decode from PSP memory.
		plan.replaced = nullptr;
		plan.doReplace = false;
		plan.saveTexture = false;
		decimationCounter_ = 0;
		Decimate(entry, true);
		const int sourceW = gstate.getTextureWidth(plan.baseLevelSrc);
		const int sourceH = gstate.getTextureHeight(plan.baseLevelSrc);
		const bool clipPSPSize = !g_DoubleTextureCoordinates &&
			(entry->addr < PSP_GetKernelMemoryBase() || entry->addr >= PSP_GetKernelMemoryEnd()) &&
			(sourceW > 512 || sourceH > 512);
		if (clipPSPSize) {
			entry->status |= MetalTexStatus::PSP_SIZE_CLIPPED;
		} else {
			entry->status &= ~MetalTexStatus::PSP_SIZE_CLIPPED;
		}
		plan.w = plan.createW = clipPSPSize ? std::min(sourceW, 512) : sourceW;
		plan.h = plan.createH = clipPSPSize ? std::min(sourceH, 512) : sourceH;
		plan.scaleFactor = 1;
		plan.levelsToLoad = plan.levelsToCreate = plan.maxPossibleLevels = 1;
		entry->status &= ~MetalTexStatus::IS_SCALED_OR_REPLACED;
		desc.type = plan.depth > 1 ? Draw::TextureType::LINEAR3D : Draw::TextureType::LINEAR2D;
		desc.width = plan.createW;
		desc.height = plan.createH;
		desc.depth = plan.depth;
		desc.format = plan.decodeToClut8 ? Draw::DataFormat::R8_UNORM : Draw::DataFormat::R8G8B8A8_UNORM;
		desc.mipLevels = 1;
		desc.generateMips = false;
		desc.initData.resize(1);
		compressedReplacement = false;
		level = 0;
		if (desc.width > 0 && desc.height > 0 && desc.width <= maxDimension && desc.height <= maxDimension) {
			texture = Metal::Texture::Create(manager_->Context(), desc, &error);
		} else {
			error = "Invalid Metal PSP texture fallback dimensions";
		}
	}
	if (!texture) {
		ERROR_LOG(Log::G3D, "Failed to build Metal PSP texture: %s", error.c_str());
		return;
	}
	entry->texturePtr = texture;
	if (plan.depth > 1) {
		entry->status |= MetalTexStatus::IS_3D;
	} else {
		entry->status &= ~MetalTexStatus::IS_3D;
	}
	if (desc.mipLevels == 1) {
		entry->status |= MetalTexStatus::NO_MIPS;
	} else {
		entry->status &= ~MetalTexStatus::NO_MIPS;
	}
	if (plan.doReplace) {
		entry->SetAlphaStatus(plan.replaced->AlphaStatus());
	}
}

bool TextureCacheMetal::ReadCompressedTextureDebug(id<MTLTexture> texture, int mip, int width, int height,
	GPUDebugBuffer &buffer, std::string *error) {
	auto &context = manager_->Context();
	if (!debugReadbackPipeline_) {
		Metal::CompiledShader vertex;
		vertex.entryPoint = "debugVertex";
		vertex.source = R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 debugVertex(uint id [[vertex_id]]) {
	float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
	return float4(p[id], 0, 1);
}
)";
		Metal::CompiledShader fragment;
		fragment.entryPoint = "debugFragment";
		fragment.source = R"(
#include <metal_stdlib>
using namespace metal;
fragment float4 debugFragment(float4 pos [[position]], texture2d<float> image [[texture(0)]],
	constant float2 &invSize [[buffer(0)]], constant uint &mip [[buffer(1)]]) {
	constexpr sampler nearest(filter::nearest, mip_filter::nearest, address::clamp_to_edge);
	return image.sample(nearest, pos.xy * invSize, level(float(mip)));
}
)";
		auto vs = context.CreateShader(vertex, "Metal debug texture vertex", error);
		auto fs = context.CreateShader(fragment, "Metal debug texture fragment", error);
		if (!vs || !fs) {
			return false;
		}
		MTL4RenderPipelineDescriptor *desc = [MTL4RenderPipelineDescriptor new];
		desc.vertexFunctionDescriptor = context.FunctionDescriptor(vs);
		desc.fragmentFunctionDescriptor = context.FunctionDescriptor(fs);
		desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		NSError *nativeError = nil;
		debugReadbackPipeline_ = [context.Compiler() newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:nil error:&nativeError];
		if (!debugReadbackPipeline_) {
			*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to create Metal debug texture pipeline";
			return false;
		}
	}
	Draw::AutoRef<Metal::Framebuffer> scratch;
	scratch.reset(Metal::Framebuffer::Create(context.Device(), {width, height, 1, 1, 0, false, "Metal debug texture"}, error));
	if (!scratch || (!context.Commands() && !context.BeginCommands(error))) {
		return false;
	}
	MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
	scratch->SetRenderAttachments(pass);
	pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
	context.ProfileRenderPass(pass, "Debug texture readback");
	auto encoder = context.RenderEncoder(context.Commands(), pass);
	if (!encoder) {
		*error = "Failed to begin Metal debug texture pass";
		return false;
	}
	const float invSize[2] = {1.0f / width, 1.0f / height};
	const uint32_t sourceMip = mip;
	context.Keep(debugReadbackPipeline_);
	[encoder setRenderPipelineState:debugReadbackPipeline_];
	context.SetTexture(encoder, MTLRenderStageFragment, texture, 0);
	context.SetBytes(encoder, MTLRenderStageFragment, invSize, sizeof(invSize), 0);
	context.SetBytes(encoder, MTLRenderStageFragment, &sourceMip, sizeof(sourceMip), 1);
	[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	context.EndEncoder(encoder);
	return Metal::Readback(context, scratch->Color(), Draw::Aspect::COLOR_BIT, 0, 0, width, height,
		Draw::DataFormat::R8G8B8A8_UNORM, buffer.GetData(), width, error);
}

bool TextureCacheMetal::GetCurrentTextureDebug(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) {
	*isFramebuffer = false;
	if (!manager_) {
		return false;
	}
	MetalTextureApplyResult result = ApplyTexture(false);
	if (result.framebuffer) {
		*isFramebuffer = true;
		return level == 0 && GetFramebufferTextureDebug(result.framebuffer, result.framebufferTextureChannel, buffer);
	}
	auto texture = (__bridge id<MTLTexture>)GetNativeTextureView(result.texCacheEntry, true);
	const bool volume = texture.textureType == MTLTextureType3D;
	if (!texture || level < 0 || (NSUInteger)level >= (volume ? texture.depth : texture.mipmapLevelCount)) {
		return false;
	}
	// The debugger's PSP level selects a slice when the equal-size levels form a volume.
	const int mip = volume ? 0 : level;
	const int width = std::max<NSUInteger>(1, texture.width >> mip);
	const int height = std::max<NSUInteger>(1, texture.height >> mip);
	const bool r8 = texture.pixelFormat == MTLPixelFormatR8Unorm;
	buffer.Allocate(width, height, r8 ? GPU_DBG_FORMAT_8BIT : GPU_DBG_FORMAT_8888);
	manager_->EndRenderPass();
	std::string error;
	const auto native = result.texCacheEntry ? static_cast<Metal::Texture *>(result.texCacheEntry->texturePtr) : nullptr;
	const bool compressed = native && Metal::IsBlockCompressed(native->Format(), nullptr);
	bool success = compressed && !volume ? ReadCompressedTextureDebug(texture, mip, width, height, buffer, &error) :
		Metal::Readback(manager_->Context(), texture, Draw::Aspect::COLOR_BIT, 0, 0, width, height,
			r8 ? Draw::DataFormat::R8_UNORM : Draw::DataFormat::R8G8B8A8_UNORM, buffer.GetData(), width, &error, mip, volume ? level : 0);
	if (!success) {
		ERROR_LOG(Log::G3D, "Metal texture debug readback: %s", error.c_str());
	}
	return success;
}
