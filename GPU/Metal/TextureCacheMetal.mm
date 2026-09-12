// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cfloat>

#include "Common/Data/Convert/ColorConv.h"
#include "Common/GPU/Metal/MetalResources.h"
#include "GPU/Metal/ShaderManagerMetal.h"
#include "GPU/Metal/TextureCacheMetal.h"
#include "Core/Config.h"
#include "Common/File/VFS/VFS.h"
#include "Common/StringUtils.h"
#include "GPU/Common/PostShader.h"

bool TextureScalerMetal::Configure(Metal::RenderContext &context, const TextureShaderInfo &info, std::string *error) {
	Clear();
	error->clear();
	if (!info.metalSupported || info.computeShaderFiles.size() != 1 || !info.constantBuffer.empty() || info.scaleFactor < 2 || info.scaleFactor >= 8) {
		*error = "Unsupported Metal texture scaling shader";
		return false;
	}
	size_t size = 0;
	uint8_t *data = g_VFS.ReadFile(info.computeShaderFiles[0].c_str(), &size);
	if (!data) {
		*error = "Failed to read Metal texture scaling shader";
		return false;
	}
	std::string kernel((const char *)data, size);
	delete[] data;
	std::string source = StringFromFormat(uploadShader, kernel.c_str());
	const std::string pushConstants = "layout(push_constant) uniform Params";
	const size_t pos = source.find(pushConstants);
	if (pos == std::string::npos) {
		*error = "Texture scaling uniform layout is missing";
		return false;
	}
	source.replace(pos, pushConstants.size(), "layout(std140, set=0, binding=2) uniform Params");
	Metal::ShaderCompileOptions options;
	options.textureBindingBase = 0;
	options.noInlineFunctions = info.metalNoInlineFunctions;
	Metal::CompiledShader shader;
	if (!Metal::CompileShader(source, ShaderStage::Compute, options, &shader, error)) {
		return false;
	}
	// Only the shared single-pass wrapper's bindings are populated below.
	for (const auto &resource : shader.resources) {
		if (!((resource.kind == Metal::ResourceKind::StorageTexture && resource.index == 0) ||
			(resource.kind == Metal::ResourceKind::StorageBuffer && resource.index == 1) ||
			(resource.kind == Metal::ResourceKind::UniformBuffer && resource.index == 2 && resource.byteSize <= 16))) {
			*error = "Unsupported Metal texture scaling resource binding";
			return false;
		}
	}
	if (shader.workgroupSize[0] != 8 || shader.workgroupSize[1] != 8 || shader.workgroupSize[2] != 1) {
		*error = "Unsupported Metal texture scaling workgroup size";
		return false;
	}
	auto function = context.CreateShader(shader, info.name.c_str(), error);
	if (!function) {
		return false;
	}
	NSError *nativeError = nil;
	pipeline_ = [context.Device() newComputePipelineStateWithFunction:function error:&nativeError];
	if (!pipeline_ || pipeline_.maxTotalThreadsPerThreadgroup < 64) {
		*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to create Metal texture scaling pipeline";
		Clear();
		return false;
	}
	scaleFactor_ = info.scaleFactor;
	return true;
}

Metal::Texture *TextureScalerMetal::Scale(Metal::RenderContext &context, const uint32_t *pixels, int width, int height, int mipLevels, std::string *error) {
	error->clear();
	if (!pipeline_ || !pixels || width <= 0 || height <= 0 || width > 16384 / scaleFactor_ || height > 16384 / scaleFactor_) {
		*error = "Invalid Metal texture scaling input";
		return nullptr;
	}
	if (!context.Commands() && !context.BeginCommands(error)) {
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
	auto input = context.Upload(pixels, (size_t)width * height * sizeof(uint32_t), error);
	if (!commands || !input) {
		texture->Release();
		return nullptr;
	}
	auto encoder = [commands computeCommandEncoder];
	if (!encoder) {
		*error = "Failed to create Metal texture scaling encoder";
		texture->Release();
		return nullptr;
	}
	const int params[4] = { width, height, 0, 0 };
	[encoder setComputePipelineState:pipeline_];
	[encoder setBuffer:input.buffer offset:input.offset atIndex:1];
	[encoder setBytes:params length:sizeof(params) atIndex:2];
	[encoder setTexture:texture->Native() atIndex:0];
	[encoder dispatchThreadgroups:MTLSizeMake((width + 7) / 8, (height + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
	[encoder endEncoding];
	if (mipLevels > 1) {
		auto blit = [commands blitCommandEncoder];
		[blit generateMipmapsForTexture:texture->Native()];
		[blit endEncoding];
	}
	return texture;
}

void TextureCacheMetal::UpdateScalingShader() {
	const std::string name = g_Config.bTexHardwareScaling ? g_Config.sTextureShaderName : "";
	if (name == scalingShader_) {
		return;
	}
	scalingShader_ = name;
	textureScaler_.Clear();
	shaderScaleFactor_ = 0;
	if (!manager_ || name.empty() || name == "Off") {
		return;
	}
	ReloadAllPostShaderInfo(draw_);
	const TextureShaderInfo *info = GetTextureShaderInfo(name);
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
	// Bias is a shader uniform on Metal, not part of MTLSamplerDescriptor. Include
	// the actual anisotropy setting since the shared key only has an enable bit.
	SamplerCacheKey nativeKey = key;
	nativeKey.lodBias = 0;
	const int aniso = key.aniso && key.minFilt && key.magFilt ? 1 << std::clamp(g_Config.iAnisotropyLevel, 0, 4) : 1;
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
	desc.mipFilter = !key.mipEnable ? MTLSamplerMipFilterNotMipmapped : key.mipFilt ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
	desc.lodMinClamp = key.mipEnable ? std::max(0.0f, key.minLevel / 256.0f) : 0.0f;
	desc.lodMaxClamp = !key.mipEnable ? 0.0f : key.maxLevel == 9 * 256 ? FLT_MAX : std::max(desc.lodMinClamp, key.maxLevel / 256.0f);
	desc.maxAnisotropy = aniso;
	auto sampler = [device newSamplerStateWithDescriptor:desc];
	if (!sampler) {
		*error = "Failed to create Metal PSP sampler";
		return nil;
	}
	cache_.emplace(cacheKey, sampler);
	return sampler;
}

TextureCacheMetal::TextureCacheMetal(Draw::DrawContext *draw, Draw2D *draw2D)
	: TextureCacheCommon(draw, draw2D) {
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
	TextureCacheCommon::DeviceLost();
	ForgetLastTexture();
	samplers_.Clear();
	textureScaler_.Clear();
	scalingShader_.clear();
	shaderScaleFactor_ = 0;
	manager_ = nullptr;
	draw_ = nullptr;
}

void TextureCacheMetal::DeviceRestore(Draw::DrawContext *draw) {
	TextureCacheCommon::DeviceRestore(draw);
	manager_ = (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
	samplers_.Clear();
	ForgetLastTexture();
	UpdateScalingShader();
}

void TextureCacheMetal::NotifyConfigChanged() {
	TextureCacheCommon::NotifyConfigChanged();
	samplers_.Clear();
	UpdateScalingShader();
}

void TextureCacheMetal::ForgetLastTexture() {
	if (draw_ && manager_) {
		for (int i = 0; i < 3; ++i) {
			draw_->BindNativeTexture(i, nullptr);
			manager_->SetNativeSampler(i, nil);
		}
	}
}

void TextureCacheMetal::BindTexture(TexCacheEntry *entry) {
	if (draw_) {
		draw_->BindNativeTexture(DRAW_BINDING_TEXTURE, GetNativeTextureView(entry, false));
	}
}

void TextureCacheMetal::Unbind() {
	if (draw_ && manager_) {
		draw_->BindNativeTexture(DRAW_BINDING_TEXTURE, nullptr);
		manager_->SetNativeSampler(DRAW_BINDING_TEXTURE, nil);
	}
}

void TextureCacheMetal::ReleaseTexture(TexCacheEntry *entry, bool deleteThem) {
	if (entry->texturePtr) {
		static_cast<Metal::Texture *>(entry->texturePtr)->Release();
		entry->texturePtr = nullptr;
	}
}

void *TextureCacheMetal::GetNativeTextureView(const TexCacheEntry *entry, bool flat) const {
	return entry && entry->texturePtr ? (__bridge void *)static_cast<Metal::Texture *>(entry->texturePtr)->Native() : nullptr;
}

void TextureCacheMetal::ApplySamplerByKey(const SamplerCacheKey &key) {
	if (!manager_) {
		return;
	}
	std::string error;
	auto sampler = samplers_.GetOrCreate(manager_->Context().Device(), key, &error);
	if (!sampler) {
		ERROR_LOG(Log::G3D, "%s", error.c_str());
	}
	manager_->SetNativeSampler(DRAW_BINDING_TEXTURE, sampler);
	if (shaderManager_) {
		static_cast<ShaderManagerMetal *>(shaderManager_)->SetSamplerLodBias(key.mipEnable ? key.lodBias / 256.0f : 0.0f);
	}
}

void TextureCacheMetal::BindAsClutTexture(Draw::Texture *texture, bool smooth) {
	if (!manager_) {
		return;
	}
	draw_->BindTexture(DRAW_BINDING_DEPAL_TEXTURE, texture);
	SamplerCacheKey key{};
	key.sClamp = true;
	key.tClamp = true;
	key.minFilt = smooth;
	key.magFilt = smooth;
	std::string error;
	manager_->SetNativeSampler(DRAW_BINDING_DEPAL_TEXTURE, samplers_.GetOrCreate(manager_->Context().Device(), key, &error));
	if (!error.empty()) {
		ERROR_LOG(Log::G3D, "%s", error.c_str());
	}
}

void TextureCacheMetal::BuildTexture(TexCacheEntry *entry) {
	BuildTexturePlan plan;
	plan.hardwareScaling = g_Config.bTexHardwareScaling && textureScaler_.ScaleFactor() > 1;
	plan.slowScaler = !plan.hardwareScaling;
	if (!manager_ || !PrepareBuildTexture(plan, entry)) {
		return;
	}
	if (plan.hardwareScaling && plan.scaleFactor > 1) {
		// Decode through the common path so CLUT expansion, swizzling, alpha
		// tracking and texture dumping retain their existing semantics.
		BuildTexturePlan decodePlan = plan;
		decodePlan.scaleFactor = 1;
		const int width = gstate.getTextureWidth(plan.baseLevelSrc);
		const int height = gstate.getTextureHeight(plan.baseLevelSrc);
		std::vector<uint32_t> pixels((size_t)width * height);
		LoadTextureLevel(*entry, (uint8_t *)pixels.data(), pixels.size() * sizeof(uint32_t), width * 4,
			decodePlan, plan.baseLevelSrc, Draw::DataFormat::R8G8B8A8_UNORM, TexDecodeFlags{});
		std::string error;
		const int levels = std::min(plan.levelsToCreate, plan.maxPossibleLevels);
		auto texture = textureScaler_.Scale(manager_->Context(), pixels.data(), width, height, levels, &error);
		if (texture) {
			ReleaseTexture(entry, true);
			entry->texturePtr = texture;
			entry->status &= ~TexStatus::IS_3D;
			if (levels == 1) {
				entry->status |= TexStatus::NO_MIPS;
			} else {
				entry->status &= ~TexStatus::NO_MIPS;
			}
			return;
		}
		WARN_LOG(Log::G3D, "Metal texture scaling failed: %s", error.c_str());
		plan.createW /= plan.scaleFactor;
		plan.createH /= plan.scaleFactor;
		plan.scaleFactor = 1;
		entry->status &= ~TexStatus::IS_SCALED_OR_REPLACED;
	}
	if (plan.depth <= 0 || plan.levelsToLoad <= 0 || plan.createW <= 0 || plan.createH <= 0 ||
		plan.createW > 16384 || plan.createH > 16384) {
		ERROR_LOG(Log::G3D, "Unsupported Metal PSP texture dimensions or mip chain");
		return;
	}
	Draw::TextureDesc desc{};
	desc.type = plan.depth > 1 ? Draw::TextureType::LINEAR3D : Draw::TextureType::LINEAR2D;
	desc.width = plan.createW;
	desc.height = plan.createH;
	desc.depth = plan.depth;
	desc.format = plan.decodeToClut8 ? Draw::DataFormat::R8_UNORM : Draw::DataFormat::R8G8B8A8_UNORM;
	if (gstate_c.Use(GPU_USE_16BIT_FORMATS) && !plan.decodeToClut8 && !plan.doReplace && !plan.saveTexture && plan.scaleFactor == 1) {
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
	if (plan.doReplace) {
		desc.format = plan.replaced->Format();
		if (!(draw_->GetDataFormatSupport(desc.format) & Draw::FMT_TEXTURE)) {
			ERROR_LOG(Log::G3D, "Unsupported replacement texture format for Metal");
			return;
		}
	}
	desc.mipLevels = std::min(plan.levelsToCreate, plan.maxPossibleLevels);
	// Generate only when level zero is the sole supplied level. Generating a full
	// chain after loading explicit PSP mips would overwrite their distinct data.
	desc.generateMips = plan.levelsToLoad == 1 && desc.mipLevels > 1;
	if (!desc.generateMips) {
		desc.mipLevels = std::min(desc.mipLevels, plan.levelsToLoad);
	}
	if (desc.mipLevels <= 0) {
		return;
	}
	desc.tag = "PSP texture";
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
		LoadTextureLevel(*entry, data, slicePitch, pitch, plan, srcLevel, desc.format, TexDecodeFlags{});
		if (!plan.doReplace) {
			convertPacked(data, w, h, pitch);
		}
		++level;
		return true;
	};
	// thin3d initializes this new allocation before the active render buffer.
	auto texture = draw_->CreateTexture(desc);
	if (!texture) {
		ERROR_LOG(Log::G3D, "Failed to build Metal PSP texture");
		return;
	}
	ReleaseTexture(entry, true);
	entry->texturePtr = texture;
	if (plan.depth > 1) {
		entry->status |= TexStatus::IS_3D;
	} else {
		entry->status &= ~TexStatus::IS_3D;
	}
	if (desc.mipLevels == 1) {
		entry->status |= TexStatus::NO_MIPS;
	} else {
		entry->status &= ~TexStatus::NO_MIPS;
	}
	if (plan.doReplace) {
		entry->SetAlphaStatus(plan.replaced->AlphaStatus());
	}
}

bool TextureCacheMetal::GetCurrentTextureDebug(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) {
	*isFramebuffer = false;
	if (!manager_) {
		return false;
	}
	TextureApplyResult result = ApplyTexture(true);
	if (result.framebuffer) {
		*isFramebuffer = true;
		return level == 0 && GetFramebufferTextureDebug(result.framebuffer, result.framebufferTextureChannel, buffer);
	}
	auto texture = (__bridge id<MTLTexture>)GetNativeTextureView(result.texCacheEntry, false);
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
	bool success = Metal::Readback(manager_->Context(), texture, Draw::Aspect::COLOR_BIT, 0, 0, width, height,
		r8 ? Draw::DataFormat::R8_UNORM : Draw::DataFormat::R8G8B8A8_UNORM, buffer.GetData(), width, &error, mip, volume ? level : 0);
	if (!success) {
		ERROR_LOG(Log::G3D, "Metal texture debug readback: %s", error.c_str());
	}
	return success;
}
