// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/Common/TextureCacheCommon.h"
#include "Common/GPU/Metal/MetalRenderManager.h"

struct TextureShaderInfo;
namespace Metal { class Texture; }

class TextureScalerMetal {
public:
	bool Configure(Metal::RenderContext &context, const TextureShaderInfo &info, std::string *error);
	Metal::Texture *Scale(Metal::RenderContext &context, const uint32_t *pixels, int width, int height, int mipLevels, std::string *error);
	void Clear() { pipeline_ = nil; scaleFactor_ = 0; }
	int ScaleFactor() const { return scaleFactor_; }

private:
	id<MTLComputePipelineState> pipeline_ = nil;
	int scaleFactor_ = 0;
};

class SamplerCacheMetal {
public:
	id<MTLSamplerState> GetOrCreate(id<MTLDevice> device, const SamplerCacheKey &key, std::string *error);
	void Clear() { cache_.clear(); }
	int Size() const { return (int)cache_.size(); }

private:
	std::map<std::pair<uint64_t, int>, id<MTLSamplerState>> cache_;
};

class TextureCacheMetal : public TextureCacheCommon {
public:
	TextureCacheMetal(Draw::DrawContext *draw, Draw2D *draw2D);
	~TextureCacheMetal() override;
	void SetFramebufferManager(FramebufferManagerCommon *manager) { framebufferManager_ = manager; }
	void ForgetLastTexture() override;
	void DeviceLost() override;
	void DeviceRestore(Draw::DrawContext *draw) override;
	void NotifyConfigChanged() override;
	bool GetCurrentTextureDebug(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) override;
	void *GetNativeTextureView(const TexCacheEntry *entry, bool flat) const override;

protected:
	void BindTexture(TexCacheEntry *entry) override;
	void Unbind() override;
	void BuildTexture(TexCacheEntry *entry) override;
	void ReleaseTexture(TexCacheEntry *entry, bool deleteThem) override;
	void BindAsClutTexture(Draw::Texture *texture, bool smooth) override;
	void ApplySamplerByKey(const SamplerCacheKey &key) override;

private:
	void UpdateScalingShader();
	Metal::RenderManager *manager_ = nullptr;
	SamplerCacheMetal samplers_;
	TextureScalerMetal textureScaler_;
	std::string scalingShader_;
};
