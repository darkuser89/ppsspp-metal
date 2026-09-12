// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>

#include "Common/GPU/Metal/MetalRenderContext.h"
#include "Common/GPU/thin3d.h"

namespace Metal {

MTLPixelFormat PixelFormat(Draw::DataFormat format);
MTLVertexFormat VertexFormat(Draw::DataFormat format);
uint32_t FormatSupport(id<MTLDevice> device, Draw::DataFormat format);

class Buffer final : public Draw::Buffer {
public:
	static Buffer *Create(id<MTLDevice> device, size_t size);
	// Rename on every write so commands already encoded against the old buffer
	// continue to see their original bytes, including partial updates.
	bool Update(id<MTLDevice> device, const uint8_t *data, size_t offset, size_t size);
	id<MTLBuffer> Native() const { return buffer_; }

private:
	id<MTLBuffer> buffer_ = nil;
};

class Texture final : public Draw::Texture {
public:
	static Texture *Create(RenderContext &context, const Draw::TextureDesc &desc, std::string *error, bool storage = false);
	bool Update(RenderContext &context, const uint8_t **data, Draw::TextureCallback callback, int levels, std::string *error);
	id<MTLTexture> Native() const { return texture_; }

private:
	bool Upload(RenderContext &context, const uint8_t **data, Draw::TextureCallback callback, int levels,
		bool initialize, std::string *error);
	id<MTLTexture> texture_ = nil;
};

class Framebuffer final : public Draw::Framebuffer {
public:
	static Framebuffer *Create(id<MTLDevice> device, const Draw::FramebufferDesc &desc, std::string *error);
	static Framebuffer *Wrap(id<MTLTexture> color, id<MTLTexture> depthStencil, std::string *error);
	void UpdateTag(const char *tag) override;
	const char *Tag() const override { return tag_.c_str(); }
	id<MTLTexture> Color() const { return color_; }
	id<MTLTexture> DepthStencil() const { return depthStencil_; }
	id<MTLTexture> ColorAttachment() const { return multisampleColor_ ? multisampleColor_ : color_; }
	id<MTLTexture> DepthStencilAttachment() const { return multisampleDepthStencil_ ? multisampleDepthStencil_ : depthStencil_; }
	int SampleCount() const { return 1 << multiSampleLevel_; }
	// Resolve at the end of each pass while retaining the individual samples
	// for subsequent draws. Color()/DepthStencil() expose the resolved images.
	void SetRenderAttachments(MTLRenderPassDescriptor *pass) const;

private:
	id<MTLTexture> color_ = nil;
	id<MTLTexture> depthStencil_ = nil;
	id<MTLTexture> multisampleColor_ = nil;
	id<MTLTexture> multisampleDepthStencil_ = nil;
	std::string tag_;
};

// Rendering-thread transfer pipelines. Copies into MSAA targets retain samples
// when the counts match, and expand resolved pixels when the counts differ.
class FramebufferCopy {
public:
	bool Copy(RenderContext &context, Framebuffer *src, int sx, int sy, Framebuffer *dst,
		int dx, int dy, int width, int height, Draw::Aspect aspects, std::string *error);

private:
	struct Pipeline {
		id<MTLRenderPipelineState> render = nil;
		id<MTLDepthStencilState> depthStencil = nil;
	};
	std::map<std::array<uint64_t, 5>, Pipeline> pipelines_;
	id<MTLFunction> vertex_ = nil;
};

// Encoders must have ended before these operations. The caller owns render-pass
// restart/invalidation. Readback submits and waits for prior work on this queue.
bool CopyImage(RenderContext &context, id<MTLTexture> src, int srcX, int srcY,
	id<MTLTexture> dst, int dstX, int dstY, int width, int height, std::string *error);
bool CopyDepthStencil(RenderContext &context, id<MTLTexture> src, int srcX, int srcY,
	id<MTLTexture> dst, int dstX, int dstY, int width, int height, Draw::Aspect aspects, std::string *error);
bool Readback(RenderContext &context, id<MTLTexture> texture, Draw::Aspect aspect,
	int x, int y, int width, int height, Draw::DataFormat format, void *pixels, int pixelStride, std::string *error, int level = 0, int z = 0);

}  // namespace Metal
