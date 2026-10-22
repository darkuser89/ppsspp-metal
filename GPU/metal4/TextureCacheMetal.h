// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <list>

#include "GPU/metal4/TextureCachePolicyMetal.h"
#include "Common/GPU/metal4/MetalRenderManager.h"
#include "Common/GPU/metal4/MetalResources.h"

struct MetalTextureShaderInfo;
struct TransformedVertex;
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
	bool ComputeSamplingRegion(const TransformedVertex *vertices, const MTLViewport &viewport, std::array<int, 4> *region) const;
	bool BindSamplingRegion(const TransformedVertex *vertices, TransformedVertex *adjusted, const std::array<int, 4> *region = nullptr);
	bool HasScaledTexture() const { return geScaledTexture_ != nullptr; }
	id<MTLSamplerState> GetSampler(const SamplerCacheKey &key, std::string *error);
	const size_t CacheSizeEstimate() const override;
	struct RegionMetrics {
		uint64_t hits = 0, misses = 0, evictions = 0, clears = 0, draws = 0, groups = 0;
		size_t sourceBytes = 0, regionBytes = 0, count = 0;
	};
	void RecordRegionDraw(bool startsGroup) {
		++regionMetrics_.draws;
		regionMetrics_.groups += startsGroup;
	}
	RegionMetrics GetRegionMetrics() const {
		auto metrics = regionMetrics_;
		metrics.sourceBytes = sourceBytes_;
		metrics.regionBytes = regionBytes_;
		metrics.count = regionCount_;
		return metrics;
	}

protected:
	void BindTexture(MetalTexCacheEntry *entry) override;
	void Unbind() override;
	void BuildTexture(MetalTexCacheEntry *entry) override;
	void ReleaseTexture(MetalTexCacheEntry *entry, bool deleteThem) override;
	void BindAsClutTexture(Draw::Texture *texture, bool smooth) override;
	void ApplySamplerByKey(const SamplerCacheKey &key) override;
	void BoundFramebufferTexture() override;

private:
	using RegionKey = std::array<int, 4>;
	using RegionLRU = std::list<std::pair<Metal::Texture *, RegionKey>>;
	struct ScaledRegion {
		Draw::AutoRef<Metal::Texture> texture;
		RegionLRU::iterator lru;
		size_t bytes;
	};
	struct ScaledTextureSource {
		int width, height;
		std::vector<uint32_t> pixels;
		std::map<RegionKey, ScaledRegion> regions;
	};
	std::map<Metal::Texture *, ScaledTextureSource> scaledSources_;
	RegionLRU regionLRU_;
	Metal::Texture *geScaledTexture_ = nullptr;
	Draw::AutoRef<Metal::Texture> temporaryRegion_;
	SamplerCacheKey geSamplerKey_{};
	size_t regionBytes_ = 0;
	size_t sourceBytes_ = 0;
	size_t regionCount_ = 0;
	RegionMetrics regionMetrics_;
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
