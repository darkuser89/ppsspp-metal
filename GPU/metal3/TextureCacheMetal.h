// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "GPU/metal3/TextureCachePolicyMetal.h"
#include "Common/GPU/metal3/MetalRenderManager.h"

struct MetalTextureShaderInfo;
namespace Metal { class Texture; }

class TextureScalerMetal {
public:
	bool Configure(Metal::RenderContext &context, const MetalTextureShaderInfo &info, std::string *error);
	Metal::Texture *Scale(Metal::RenderContext &context, Metal::UploadSlice input, int width, int height, int mipLevels, std::string *error);
	void Clear() { pipelines_.fill(nil); pipelineCount_ = 0; constants_ = nil; scaleFactor_ = 0; }
	int ScaleFactor() const { return scaleFactor_; }

private:
	std::array<id<MTLComputePipelineState>, 4> pipelines_{};
	int pipelineCount_ = 0;
	id<MTLBuffer> constants_ = nil;
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

class TextureCacheMetal : public MetalTextureCacheCommon {
public:
	TextureCacheMetal(Draw::DrawContext *draw, Draw2D *draw2D);
	~TextureCacheMetal() override;
	void SetFramebufferManager(FramebufferManagerMetal *manager) { framebufferManager_ = manager; }
	void ForgetLastTexture() override;
	void DeviceLost() override;
	void DeviceRestore(Draw::DrawContext *draw) override;
	void NotifyConfigChanged() override;
	bool GetCurrentTextureDebug(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) override;
	void *GetNativeTextureView(const MetalTexCacheEntry *entry, bool flat) const override;
	void ResetGETextureBindings();
	void RestoreGETextureBindings();

protected:
	void BindTexture(MetalTexCacheEntry *entry) override;
	void Unbind() override;
	void BuildTexture(MetalTexCacheEntry *entry) override;
	void ReleaseTexture(MetalTexCacheEntry *entry, bool deleteThem) override;
	void BindAsClutTexture(Draw::Texture *texture, bool smooth) override;
	void ApplySamplerByKey(const SamplerCacheKey &key) override;
	void BoundFramebufferTexture() override;

private:
	void UpdateScalingShader();
	bool ReadCompressedTextureDebug(id<MTLTexture> texture, int mip, int width, int height, GPUDebugBuffer &buffer, std::string *error);
	Metal::RenderManager *manager_ = nullptr;
	id<MTLTexture> missingTexture_ = nil;
	id<MTLTexture> missingTexture3D_ = nil;
	id<MTLTexture> geTexture_ = nil;
	id<MTLSamplerState> geSampler_ = nil;
	id<MTLTexture> geClutTexture_ = nil;
	id<MTLSamplerState> geClutSampler_ = nil;
	SamplerCacheMetal samplers_;
	TextureScalerMetal textureScaler_;
	id<MTLRenderPipelineState> debugReadbackPipeline_ = nil;
	std::string scalingShader_;
};
