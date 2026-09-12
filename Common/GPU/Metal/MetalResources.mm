// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "Common/GPU/Metal/MetalResources.h"
#include "Common/Data/Convert/ColorConv.h"
#include "Common/StringUtils.h"

namespace Metal {

MTLPixelFormat PixelFormat(Draw::DataFormat format) {
	using Draw::DataFormat;
	switch (format) {
	case DataFormat::R8_UNORM: return MTLPixelFormatR8Unorm;
	case DataFormat::R8G8_UNORM: return MTLPixelFormatRG8Unorm;
	// Metal names these from the least significant component, unlike DataFormat.
	case DataFormat::R5G6B5_UNORM_PACK16: return MTLPixelFormatB5G6R5Unorm;
	case DataFormat::R5G5B5A1_UNORM_PACK16: return MTLPixelFormatA1BGR5Unorm;
	case DataFormat::R4G4B4A4_UNORM_PACK16: return MTLPixelFormatABGR4Unorm;
	case DataFormat::R8G8B8A8_UNORM: return MTLPixelFormatRGBA8Unorm;
	case DataFormat::B8G8R8A8_UNORM: return MTLPixelFormatBGRA8Unorm;
	case DataFormat::R8G8B8A8_UNORM_SRGB: return MTLPixelFormatRGBA8Unorm_sRGB;
	case DataFormat::B8G8R8A8_UNORM_SRGB: return MTLPixelFormatBGRA8Unorm_sRGB;
	case DataFormat::R16_UNORM: return MTLPixelFormatR16Unorm;
	case DataFormat::R16_FLOAT: return MTLPixelFormatR16Float;
	case DataFormat::R16G16_FLOAT: return MTLPixelFormatRG16Float;
	case DataFormat::R16G16B16A16_FLOAT: return MTLPixelFormatRGBA16Float;
	case DataFormat::R32_FLOAT: return MTLPixelFormatR32Float;
	case DataFormat::D16: return MTLPixelFormatDepth16Unorm;
	case DataFormat::D32F: return MTLPixelFormatDepth32Float;
	case DataFormat::D32F_S8: return MTLPixelFormatDepth32Float_Stencil8;
	default: return MTLPixelFormatInvalid;
	}
}

MTLVertexFormat VertexFormat(Draw::DataFormat format) {
	using Draw::DataFormat;
	switch (format) {
	case DataFormat::R32_FLOAT: return MTLVertexFormatFloat;
	case DataFormat::R32G32_FLOAT: return MTLVertexFormatFloat2;
	case DataFormat::R32G32B32_FLOAT: return MTLVertexFormatFloat3;
	case DataFormat::R32G32B32A32_FLOAT: return MTLVertexFormatFloat4;
	case DataFormat::R8G8B8A8_UNORM: return MTLVertexFormatUChar4Normalized;
	case DataFormat::R8G8B8A8_SNORM: return MTLVertexFormatChar4Normalized;
	case DataFormat::R8G8B8A8_UINT: return MTLVertexFormatUChar4;
	case DataFormat::R8G8B8A8_SINT: return MTLVertexFormatChar4;
	case DataFormat::R16G16_FLOAT: return MTLVertexFormatHalf2;
	case DataFormat::R16G16B16A16_FLOAT: return MTLVertexFormatHalf4;
	default: return MTLVertexFormatInvalid;
	}
}

uint32_t FormatSupport(id<MTLDevice> device, Draw::DataFormat format) {
	uint32_t support = VertexFormat(format) != MTLVertexFormatInvalid ? Draw::FMT_INPUTLAYOUT : 0;
	if (format == Draw::DataFormat::R5G6B5_UNORM_PACK16 || format == Draw::DataFormat::R5G5B5A1_UNORM_PACK16 ||
		format == Draw::DataFormat::R4G4B4A4_UNORM_PACK16) {
		// Packed color formats are Apple GPU features, not part of the Mac2 baseline.
		return [device supportsFamily:MTLGPUFamilyApple2] ? Draw::FMT_TEXTURE | Draw::FMT_RENDERTARGET | Draw::FMT_AUTOGEN_MIPS : 0;
	}
	if (Draw::DataFormatIsDepthStencil(format)) {
		// Framebuffers allocate D32F_S8. Standalone depth textures and other
		// depth attachment formats aren't exposed by this resource path yet.
		return format == Draw::DataFormat::D32F_S8 ? Draw::FMT_DEPTHSTENCIL : 0;
	}
	if (PixelFormat(format) != MTLPixelFormatInvalid) {
		support |= Draw::FMT_TEXTURE | Draw::FMT_RENDERTARGET;
		// R32Float isn't filterable on every supported Metal 3 device.
		if (format != Draw::DataFormat::R32_FLOAT) {
			support |= Draw::FMT_AUTOGEN_MIPS;
		}
	}
	return support;
}

Buffer *Buffer::Create(id<MTLDevice> device, size_t size) {
	if (!size) {
		return nullptr;
	}
	if (@available(macOS 13.0, iOS 16.0, *)) {
		if (size > device.maxBufferLength) {
			return nullptr;
		}
	} else {
		return nullptr;
	}
	id<MTLBuffer> native = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (!native) {
		return nullptr;
	}
	auto buffer = new Buffer();
	buffer->buffer_ = native;
	return buffer;
}

bool Buffer::Update(id<MTLDevice> device, const uint8_t *data, size_t offset, size_t size) {
	if (offset > buffer_.length || size > buffer_.length - offset || (size && !data)) {
		return false;
	}
	if (!size) {
		return true;
	}
	id<MTLBuffer> replacement = [device newBufferWithLength:buffer_.length options:MTLResourceStorageModeShared];
	if (!replacement) {
		return false;
	}
	memcpy(replacement.contents, buffer_.contents, buffer_.length);
	memcpy((uint8_t *)replacement.contents + offset, data, size);
	buffer_ = replacement;
	return true;
}

static bool EnsureCommands(RenderContext &context, std::string *error) {
	return context.Commands() || context.BeginCommands(error);
}

Texture *Texture::Create(RenderContext &context, const Draw::TextureDesc &desc, std::string *error, bool storage) {
	error->clear();
	const auto format = PixelFormat(desc.format);
	const bool volume = desc.type == Draw::TextureType::LINEAR3D;
	const int maxDimension = volume ? 2048 : 16384;
	if ((!volume && desc.type != Draw::TextureType::LINEAR2D) || (!volume && desc.depth != 1) ||
		desc.depth <= 0 || desc.depth > maxDimension || desc.width <= 0 || desc.height <= 0 ||
		desc.width > maxDimension || desc.height > maxDimension || desc.mipLevels <= 0 || format == MTLPixelFormatInvalid ||
		!(FormatSupport(context.Device(), desc.format) & Draw::FMT_TEXTURE)) {
		*error = "Unsupported Metal texture type, dimensions, or upload format";
		return nullptr;
	}
	int maxLevels = 1;
	for (int side = std::max({desc.width, desc.height, desc.depth}); side > 1; side >>= 1) {
		++maxLevels;
	}
	if (desc.mipLevels > maxLevels || desc.initData.size() > (size_t)desc.mipLevels ||
		(desc.generateMips && !(FormatSupport(context.Device(), desc.format) & Draw::FMT_AUTOGEN_MIPS))) {
		*error = "Invalid Metal texture mip chain";
		return nullptr;
	}
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:desc.width height:desc.height mipmapped:NO];
	td.textureType = volume ? MTLTextureType3D : MTLTextureType2D;
	td.depth = desc.depth;
	td.mipmapLevelCount = desc.mipLevels;
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageShaderRead;
	if (storage) {
		if (desc.format != Draw::DataFormat::R8G8B8A8_UNORM || volume) {
			*error = "Metal compute uploads require a 2D RGBA8 texture";
			return nullptr;
		}
		td.usage |= MTLTextureUsageShaderWrite;
	}
	if (desc.swizzle != Draw::TextureSwizzle::DEFAULT) {
		if (desc.format != Draw::DataFormat::R8_UNORM) {
			*error = "Metal texture swizzle requires R8";
			return nullptr;
		}
		if (@available(macOS 13.0, iOS 16.0, *)) {
			MTLTextureSwizzle r = MTLTextureSwizzleRed;
			MTLTextureSwizzle rgb = desc.swizzle == Draw::TextureSwizzle::R8_AS_ALPHA ? MTLTextureSwizzleOne : r;
			MTLTextureSwizzle alpha = desc.swizzle == Draw::TextureSwizzle::R8_AS_GRAYSCALE ? MTLTextureSwizzleOne : r;
			td.swizzle = MTLTextureSwizzleChannelsMake(rgb, rgb, rgb, alpha);
		} else {
			*error = "Metal 3 texture swizzles require macOS 13 or iOS 16";
			return nullptr;
		}
	}
	id<MTLTexture> native = [context.Device() newTextureWithDescriptor:td];
	if (!native) {
		*error = "Failed to allocate Metal texture";
		return nullptr;
	}
	auto texture = new Texture();
	texture->texture_ = native;
	texture->width_ = desc.width;
	texture->height_ = desc.height;
	texture->depth_ = desc.depth;
	texture->format_ = desc.format;
	if (desc.tag) {
		native.label = [NSString stringWithUTF8String:desc.tag];
	}
	// Callbacks can generate data without an initData pointer.
	std::vector<const uint8_t *> data = desc.initData;
	if (data.empty() && desc.initDataCallback) {
		data.resize(desc.generateMips ? 1 : desc.mipLevels, nullptr);
	}
	if ((!data.empty() && !texture->Upload(context, data.data(), desc.initDataCallback, (int)data.size(), true, error)) ||
		(desc.generateMips && data.empty())) {
		if (error->empty()) {
			*error = "Mipmap generation needs initialized base data";
		}
		texture->Release();
		return nullptr;
	}
	if (desc.generateMips && desc.mipLevels > 1) {
		if (!EnsureCommands(context, error)) {
			texture->Release();
			return nullptr;
		}
		auto commands = context.InitializationCommands(error);
		if (!commands) {
			texture->Release();
			return nullptr;
		}
		id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
		[blit generateMipmapsForTexture:native];
		[blit endEncoding];
	}
	return texture;
}

bool Texture::Update(RenderContext &context, const uint8_t **data, Draw::TextureCallback callback, int levels, std::string *error) {
	return Upload(context, data, callback, levels, false, error);
}

bool Texture::Upload(RenderContext &context, const uint8_t **data, Draw::TextureCallback callback, int levels,
	bool initialize, std::string *error) {
	error->clear();
	if (levels < 0 || (size_t)levels > texture_.mipmapLevelCount || (levels && !data)) {
		*error = "Invalid Metal texture upload levels";
		return false;
	}
	// Prepare the complete upload first, before changing the texture or opening an encoder.
	std::vector<id<MTLBuffer>> staging;
	for (int level = 0; level < levels; ++level) {
		const int w = std::max(1, width_ >> level);
		const int h = std::max(1, height_ >> level);
		const int d = std::max(1, depth_ >> level);
		const size_t rowBytes = w * Draw::DataFormatSizeInBytes(format_);
		const size_t pitch = (rowBytes + 255) & ~size_t(255);
		const size_t slicePitch = pitch * h;
		if (@available(macOS 13.0, iOS 16.0, *)) {
			if (slicePitch * d > context.Device().maxBufferLength) {
				*error = "Metal texture upload exceeds the device buffer limit";
				return false;
			}
		} else {
			*error = "Metal 3 texture uploads require macOS 13 or iOS 16";
			return false;
		}
		id<MTLBuffer> upload = [context.Device() newBufferWithLength:slicePitch * d options:MTLResourceStorageModeShared];
		if (!upload) {
			*error = "Failed to allocate Metal texture staging buffer";
			return false;
		}
		bool generated = callback && callback((uint8_t *)upload.contents, data[level], w, h, d, (uint32_t)pitch, (uint32_t)slicePitch);
		// Thin3d callbacks return false to request a copy from the original data.
		if (!generated) {
			if (!data[level]) {
				*error = "Missing Metal texture mip data";
				return false;
			}
			for (int z = 0; z < d; ++z) {
				for (int y = 0; y < h; ++y) {
					memcpy((uint8_t *)upload.contents + slicePitch * z + pitch * y, data[level] + rowBytes * (h * z + y), rowBytes);
				}
			}
		}
		staging.push_back(upload);
	}
	if (levels == 0) {
		return true;
	}
	if (!EnsureCommands(context, error)) {
		return false;
	}
	auto commands = initialize ? context.InitializationCommands(error) : context.Commands();
	if (!commands) {
		return false;
	}
	id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
	for (int level = 0; level < levels; ++level) {
		const int w = std::max(1, width_ >> level);
		const int h = std::max(1, height_ >> level);
		const size_t pitch = (w * Draw::DataFormatSizeInBytes(format_) + 255) & ~size_t(255);
		[blit copyFromBuffer:staging[level] sourceOffset:0 sourceBytesPerRow:pitch sourceBytesPerImage:pitch * h
			sourceSize:MTLSizeMake(w, h, std::max(1, depth_ >> level)) toTexture:texture_ destinationSlice:0 destinationLevel:level destinationOrigin:MTLOriginMake(0, 0, 0)];
	}
	[blit endEncoding];
	return true;
}

Framebuffer *Framebuffer::Create(id<MTLDevice> device, const Draw::FramebufferDesc &desc, std::string *error) {
	error->clear();
	if (desc.width <= 0 || desc.height <= 0 || desc.width > 16384 || desc.height > 16384 ||
		desc.numLayers != 1 || desc.multiSampleLevel < 0 || desc.multiSampleLevel > 4 ||
		![device supportsTextureSampleCount:1u << desc.multiSampleLevel]) {
		*error = "Unsupported Metal framebuffer dimensions, layers, or sample count";
		return nullptr;
	}
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:desc.width height:desc.height mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	id<MTLTexture> color = [device newTextureWithDescriptor:td];
	id<MTLTexture> depthStencil = nil;
	if (desc.z_stencil) {
		td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
		depthStencil = [device newTextureWithDescriptor:td];
	}
	id<MTLTexture> multisampleColor = nil;
	id<MTLTexture> multisampleDepthStencil = nil;
	if (desc.multiSampleLevel > 0) {
		td.textureType = MTLTextureType2DMultisample;
		td.sampleCount = 1u << desc.multiSampleLevel;
		td.pixelFormat = MTLPixelFormatRGBA8Unorm;
		multisampleColor = [device newTextureWithDescriptor:td];
		if (desc.z_stencil) {
			td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
			multisampleDepthStencil = [device newTextureWithDescriptor:td];
		}
	}
	if (!color || (desc.z_stencil && !depthStencil) ||
		(desc.multiSampleLevel > 0 && (!multisampleColor || (desc.z_stencil && !multisampleDepthStencil)))) {
		*error = "Failed to allocate Metal framebuffer";
		return nullptr;
	}
	auto fbo = new Framebuffer();
	fbo->color_ = color;
	fbo->depthStencil_ = depthStencil;
	fbo->width_ = desc.width;
	fbo->multisampleColor_ = multisampleColor;
	fbo->multisampleDepthStencil_ = multisampleDepthStencil;
	fbo->height_ = desc.height;
	fbo->layers_ = 1;
	fbo->multiSampleLevel_ = desc.multiSampleLevel;
	fbo->UpdateTag(desc.tag);
	return fbo;
}

void Framebuffer::UpdateTag(const char *tag) {
	tag_ = tag ? tag : "Framebuffer";
	color_.label = [NSString stringWithUTF8String:tag_.c_str()];
	depthStencil_.label = [NSString stringWithUTF8String:(tag_ + " depth/stencil").c_str()];
	multisampleColor_.label = [NSString stringWithUTF8String:(tag_ + " MSAA color").c_str()];
	multisampleDepthStencil_.label = [NSString stringWithUTF8String:(tag_ + " MSAA depth/stencil").c_str()];
}

Framebuffer *Framebuffer::Wrap(id<MTLTexture> color, id<MTLTexture> depthStencil, std::string *error) {
	error->clear();
	if (!color || color.textureType != MTLTextureType2D || color.sampleCount != 1 ||
		(color.pixelFormat != MTLPixelFormatRGBA8Unorm && color.pixelFormat != MTLPixelFormatBGRA8Unorm) ||
		(depthStencil && (depthStencil.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
		depthStencil.width != color.width || depthStencil.height != color.height || depthStencil.sampleCount != 1))) {
		*error = "Invalid Metal framebuffer attachment textures";
		return nullptr;
	}
	auto fbo = new Framebuffer();
	fbo->color_ = color;
	fbo->depthStencil_ = depthStencil;
	fbo->width_ = (int)color.width;
	fbo->height_ = (int)color.height;
	fbo->layers_ = 1;
	fbo->multiSampleLevel_ = 0;
	fbo->tag_ = "Metal surface";
	return fbo;
}

void Framebuffer::SetRenderAttachments(MTLRenderPassDescriptor *pass) const {
	pass.colorAttachments[0].texture = ColorAttachment();
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	if (multisampleColor_) {
		pass.colorAttachments[0].resolveTexture = color_;
		pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
	}
	if (depthStencil_) {
		pass.depthAttachment.texture = DepthStencilAttachment();
		pass.stencilAttachment.texture = DepthStencilAttachment();
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.stencilAttachment.storeAction = MTLStoreActionStore;
		if (multisampleDepthStencil_) {
			if (@available(macOS 13.0, iOS 16.0, *)) {
				pass.depthAttachment.resolveTexture = depthStencil_;
				pass.stencilAttachment.resolveTexture = depthStencil_;
				pass.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
				pass.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
				pass.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
				pass.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
			}
		}
	}
}

static bool ValidRect(id<MTLTexture> texture, int x, int y, int w, int h) {
	return texture && (texture.textureType == MTLTextureType2D || texture.textureType == MTLTextureType2DMultisample) &&
		x >= 0 && y >= 0 && w > 0 && h > 0 &&
		(size_t)x <= texture.width && (size_t)y <= texture.height &&
		(size_t)w <= texture.width - x && (size_t)h <= texture.height - y;
}

bool CopyImage(RenderContext &context, id<MTLTexture> src, int srcX, int srcY,
	id<MTLTexture> dst, int dstX, int dstY, int width, int height, std::string *error) {
	error->clear();
	if (!ValidRect(src, srcX, srcY, width, height) || !ValidRect(dst, dstX, dstY, width, height) ||
		src.pixelFormat != dst.pixelFormat || src.sampleCount != dst.sampleCount) {
		*error = "Invalid Metal image copy region or format";
		return false;
	}
	if (!EnsureCommands(context, error)) {
		return false;
	}
	// Metal does not permit overlapping copies within one texture. Snapshot the
	// region first; this also makes framebuffer feedback copies deterministic.
	id<MTLTexture> temporary = nil;
	if (src == dst) {
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:width height:height mipmapped:NO];
		td.textureType = src.textureType;
		td.sampleCount = src.sampleCount;
		td.storageMode = MTLStorageModePrivate;
		temporary = [context.Device() newTextureWithDescriptor:td];
		if (!temporary) {
			*error = "Failed to allocate Metal copy snapshot";
			return false;
		}
	}
	id<MTLBlitCommandEncoder> blit = [context.Commands() blitCommandEncoder];
	if (temporary) {
		[blit copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(srcX, srcY, 0)
			sourceSize:MTLSizeMake(width, height, 1) toTexture:temporary destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
		src = temporary;
		srcX = srcY = 0;
	}
	[blit copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(srcX, srcY, 0)
		sourceSize:MTLSizeMake(width, height, 1) toTexture:dst destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(dstX, dstY, 0)];
	[blit endEncoding];
	return true;
}

bool CopyDepthStencil(RenderContext &context, id<MTLTexture> src, int srcX, int srcY,
	id<MTLTexture> dst, int dstX, int dstY, int width, int height, Draw::Aspect aspects, std::string *error) {
	error->clear();
	if (!ValidRect(src, srcX, srcY, width, height) || !ValidRect(dst, dstX, dstY, width, height) ||
		src.sampleCount != 1 || dst.sampleCount != 1 ||
		src.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 || dst.pixelFormat != src.pixelFormat ||
		(aspects != Draw::Aspect::DEPTH_BIT && aspects != Draw::Aspect::STENCIL_BIT &&
			aspects != (Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT))) {
		*error = "Invalid Metal depth/stencil copy region, format or aspects";
		return false;
	}
	if (aspects == (Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT)) {
		return CopyImage(context, src, srcX, srcY, dst, dstX, dstY, width, height, error);
	}
	// Texture-to-texture blits copy both packed aspects. Transfer one plane
	// through a GPU buffer to preserve the other, also snapshotting self-copies.
	const bool depth = aspects == Draw::Aspect::DEPTH_BIT;
	const MTLBlitOption option = depth ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionStencilFromDepthStencil;
	const size_t pitch = ((size_t)width * (depth ? 4 : 1) + 255) & ~size_t(255);
	id<MTLBuffer> plane = [context.Device() newBufferWithLength:pitch * height options:MTLResourceStorageModePrivate];
	if (!plane || !EnsureCommands(context, error)) {
		if (error->empty()) {
			*error = "Failed to allocate Metal depth/stencil copy buffer";
		}
		return false;
	}
	id<MTLBlitCommandEncoder> blit = [context.Commands() blitCommandEncoder];
	[blit copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(srcX, srcY, 0)
		sourceSize:MTLSizeMake(width, height, 1) toBuffer:plane destinationOffset:0
		destinationBytesPerRow:pitch destinationBytesPerImage:pitch * height options:option];
	[blit copyFromBuffer:plane sourceOffset:0 sourceBytesPerRow:pitch sourceBytesPerImage:pitch * height
		sourceSize:MTLSizeMake(width, height, 1) toTexture:dst destinationSlice:0 destinationLevel:0
		destinationOrigin:MTLOriginMake(dstX, dstY, 0) options:option];
	[blit endEncoding];
	return true;
}

bool FramebufferCopy::Copy(RenderContext &context, Framebuffer *src, int sx, int sy, Framebuffer *dst,
	int dx, int dy, int width, int height, Draw::Aspect aspects, std::string *error) {
	error->clear();
	const bool color = aspects & Draw::Aspect::COLOR_BIT;
	const bool depth = aspects & Draw::Aspect::DEPTH_BIT;
	const bool stencil = aspects & Draw::Aspect::STENCIL_BIT;
	const auto allowed = Draw::Aspect::COLOR_BIT | Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT;
	if (!src || !dst || !(color || depth || stencil) || (aspects & ~allowed) ||
		!ValidRect(src->Color(), sx, sy, width, height) || !ValidRect(dst->Color(), dx, dy, width, height) ||
		((depth || stencil) && (!src->DepthStencil() || !dst->DepthStencil()))) {
		*error = "Invalid Metal framebuffer transfer region or aspects";
		return false;
	}
	// Read from a separate resource even for non-overlapping self-copies: Metal
	// forbids sampling an attachment in its own render pass.
	Draw::AutoRef<Framebuffer> snapshot;
	if (src == dst) {
		snapshot.reset(Framebuffer::Create(context.Device(), {width, height, 1, 1, src->MultiSampleLevel(), depth || stencil, "Metal transfer snapshot"}, error));
		if (!snapshot ||
			(color && !CopyImage(context, src->ColorAttachment(), sx, sy, snapshot->ColorAttachment(), 0, 0, width, height, error)) ||
			((depth || stencil) && !CopyImage(context, src->DepthStencilAttachment(), sx, sy, snapshot->DepthStencilAttachment(), 0, 0, width, height, error))) {
			return false;
		}
		src = snapshot.ptr;
		sx = sy = 0;
	}
	const bool perSample = src->SampleCount() > 1 && src->SampleCount() == dst->SampleCount();
	id<MTLTexture> srcColor = perSample ? src->ColorAttachment() : src->Color();
	id<MTLTexture> srcDepth = perSample ? src->DepthStencilAttachment() : src->DepthStencil();
	id<MTLTexture> srcStencil = stencil ? [srcDepth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8] : nil;
	if (stencil && !srcStencil) {
		*error = "Failed to create Metal stencil transfer view";
		return false;
	}
	const auto depthFormat = dst->DepthStencil() ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid;
	const std::array<uint64_t, 5> key{(uint64_t)aspects, perSample, (uint64_t)dst->SampleCount(), (uint64_t)dst->Color().pixelFormat, (uint64_t)depthFormat};
	auto found = pipelines_.find(key);
	if (found == pipelines_.end()) {
		if (!vertex_) {
			CompiledShader shader;
			shader.entryPoint = "copyVertex";
			shader.source = R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 copyVertex(uint id [[vertex_id]]) {
	float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
	return float4(p[id], 0, 1);
}
)";
			vertex_ = context.CreateShader(shader, "Metal transfer vertex", error);
			if (!vertex_) {
				return false;
			}
		}
		CompiledShader shader;
		shader.entryPoint = "copyFragment";
		shader.source = "#include <metal_stdlib>\nusing namespace metal;\nstruct Out {\n";
		if (color) {
			shader.source += "float4 color [[color(0)]];\n";
		}
		if (depth) {
			shader.source += "float depth [[depth(any)]];\n";
		}
		if (stencil) {
			shader.source += "uint stencil [[stencil]];\n";
		}
		shader.source += "};\nfragment Out copyFragment(float4 pos [[position]], constant int2 &offset [[buffer(0)]]";
		const std::string textureType = perSample ? "texture2d_ms" : "texture2d";
		if (perSample) {
			shader.source += ", uint sample [[sample_id]]";
		}
		if (color) {
			shader.source += ", " + textureType + "<float, access::read> colorTex [[texture(0)]]";
		}
		if (depth) {
			shader.source += ", " + textureType + "<float, access::read> depthTex [[texture(1)]]";
		}
		if (stencil) {
			shader.source += ", " + textureType + "<uint, access::read> stencilTex [[texture(2)]]";
		}
		shader.source += ") { uint2 p = uint2(int2(pos.xy) + offset); Out out;\n";
		const std::string read = perSample ? ".read(p, sample)" : ".read(p)";
		if (color) {
			shader.source += "out.color = colorTex" + read + ";\n";
		}
		if (depth) {
			shader.source += "out.depth = depthTex" + read + ".r;\n";
		}
		if (stencil) {
			shader.source += "out.stencil = stencilTex" + read + ".r;\n";
		}
		shader.source += "return out; }\n";
		auto fragment = context.CreateShader(shader, "Metal transfer fragment", error);
		if (!fragment) {
			return false;
		}
		MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
		pd.vertexFunction = vertex_;
		pd.fragmentFunction = fragment;
		pd.rasterSampleCount = dst->SampleCount();
		pd.colorAttachments[0].pixelFormat = dst->Color().pixelFormat;
		pd.colorAttachments[0].writeMask = color ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
		pd.depthAttachmentPixelFormat = depthFormat;
		pd.stencilAttachmentPixelFormat = depthFormat;
		NSError *nativeError = nil;
		Pipeline pipeline;
		pipeline.render = [context.Device() newRenderPipelineStateWithDescriptor:pd error:&nativeError];
		if (!pipeline.render) {
			*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to create Metal transfer pipeline";
			return false;
		}
		MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
		dd.depthCompareFunction = MTLCompareFunctionAlways;
		dd.depthWriteEnabled = depth;
		if (stencil) {
			MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
			sd.stencilCompareFunction = MTLCompareFunctionAlways;
			sd.depthStencilPassOperation = MTLStencilOperationReplace;
			dd.frontFaceStencil = sd;
			dd.backFaceStencil = sd;
		}
		pipeline.depthStencil = [context.Device() newDepthStencilStateWithDescriptor:dd];
		if (!pipeline.depthStencil) {
			*error = "Failed to create Metal transfer depth/stencil state";
			return false;
		}
		found = pipelines_.emplace(key, pipeline).first;
	}
	if (!EnsureCommands(context, error)) {
		return false;
	}
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	dst->SetRenderAttachments(rp);
	rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
	rp.depthAttachment.loadAction = MTLLoadActionLoad;
	rp.stencilAttachment.loadAction = MTLLoadActionLoad;
	auto encoder = [context.Commands() renderCommandEncoderWithDescriptor:rp];
	if (!encoder) {
		*error = "Failed to begin Metal transfer pass";
		return false;
	}
	[encoder setRenderPipelineState:found->second.render];
	[encoder setDepthStencilState:found->second.depthStencil];
	[encoder setCullMode:MTLCullModeNone];
	[encoder setViewport:(MTLViewport){(double)dx, (double)dy, (double)width, (double)height, 0, 1}];
	[encoder setScissorRect:(MTLScissorRect){(NSUInteger)dx, (NSUInteger)dy, (NSUInteger)width, (NSUInteger)height}];
	const int offset[2] = {sx - dx, sy - dy};
	[encoder setFragmentBytes:offset length:sizeof(offset) atIndex:0];
	if (color) {
		[encoder setFragmentTexture:srcColor atIndex:0];
	}
	if (depth) {
		[encoder setFragmentTexture:srcDepth atIndex:1];
	}
	if (stencil) {
		[encoder setFragmentTexture:srcStencil atIndex:2];
	}
	[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[encoder endEncoding];
	return true;
}

bool Readback(RenderContext &context, id<MTLTexture> texture, Draw::Aspect aspect,
	int x, int y, int width, int height, Draw::DataFormat format, void *pixels, int pixelStride, std::string *error, int level, int z) {
	using Draw::DataFormat;
	error->clear();
	// PSP framebuffers can have overlapping rows, including stride zero. The
	// shared converters write rows in order, matching the other backends.
	if (!pixels || pixelStride < 0 || !texture || level < 0 || (NSUInteger)level >= texture.mipmapLevelCount ||
		(texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType3D) ||
		z < 0 || (NSUInteger)z >= std::max<NSUInteger>(1, texture.depth >> level) ||
		x < 0 || y < 0 || width <= 0 || height <= 0 ||
		(uint64_t)x + width > std::max<NSUInteger>(1, texture.width >> level) ||
		(uint64_t)y + height > std::max<NSUInteger>(1, texture.height >> level)) {
		*error = StringFromFormat("Invalid Metal readback: region %d,%d %dx%d, stride %d, texture %lux%lu, mip %d/%lu, destination %s",
			x, y, width, height, pixelStride, (unsigned long)texture.width, (unsigned long)texture.height,
			level, (unsigned long)texture.mipmapLevelCount, pixels ? "present" : "null");
		return false;
	}
	size_t bytesPerPixel = 4;
	MTLBlitOption options = MTLBlitOptionNone;
	const bool rawR8 = aspect == Draw::Aspect::COLOR_BIT && texture.pixelFormat == MTLPixelFormatR8Unorm && format == DataFormat::R8_UNORM;
	const bool packed16 = aspect == Draw::Aspect::COLOR_BIT &&
		(texture.pixelFormat == MTLPixelFormatB5G6R5Unorm || texture.pixelFormat == MTLPixelFormatA1BGR5Unorm || texture.pixelFormat == MTLPixelFormatABGR4Unorm);
	const bool color = aspect == Draw::Aspect::COLOR_BIT && !rawR8 && !packed16;
	const bool stencil = aspect == Draw::Aspect::STENCIL_BIT;
	if (rawR8) {
		bytesPerPixel = 1;
	} else if (packed16) {
		if (format != DataFormat::R8G8B8A8_UNORM) {
			*error = "Packed Metal texture readback requires RGBA8 output";
			return false;
		}
		bytesPerPixel = 2;
	} else if (color) {
		if ((texture.pixelFormat != MTLPixelFormatRGBA8Unorm && texture.pixelFormat != MTLPixelFormatBGRA8Unorm) ||
			(format != DataFormat::R8G8B8A8_UNORM && format != DataFormat::R8G8B8_UNORM &&
			format != DataFormat::R5G6B5_UNORM_PACK16 && format != DataFormat::A1R5G5B5_UNORM_PACK16 && format != DataFormat::A4R4G4B4_UNORM_PACK16)) {
			*error = "Unsupported Metal color readback format";
			return false;
		}
	} else if (texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
		if (stencil && format == DataFormat::S8) {
			bytesPerPixel = 1;
			options = MTLBlitOptionStencilFromDepthStencil;
		} else if (aspect == Draw::Aspect::DEPTH_BIT && (format == DataFormat::D32F || format == DataFormat::D16)) {
			options = MTLBlitOptionDepthFromDepthStencil;
		} else {
			*error = "Unsupported Metal depth/stencil readback aspect or format";
			return false;
		}
	} else {
		*error = "Unsupported Metal readback texture format";
		return false;
	}
	const size_t pitch = (width * bytesPerPixel + 255) & ~size_t(255);
	id<MTLBuffer> readback = [context.Device() newBufferWithLength:pitch * height options:MTLResourceStorageModeShared];
	if (!readback || !EnsureCommands(context, error)) {
		if (error->empty()) {
			*error = "Failed to allocate Metal readback buffer";
		}
		return false;
	}
	id<MTLBlitCommandEncoder> blit = [context.Commands() blitCommandEncoder];
	[blit copyFromTexture:texture sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(x, y, z)
		sourceSize:MTLSizeMake(width, height, 1) toBuffer:readback destinationOffset:0
		destinationBytesPerRow:pitch destinationBytesPerImage:pitch * height options:options];
	[blit endEncoding];
	if (!context.SubmitCommands(true, error)) {
		return false;
	}
	if (packed16) {
		for (int row = 0; row < height; ++row) {
			auto dst = (uint32_t *)pixels + row * pixelStride;
			auto src = (const uint16_t *)((const uint8_t *)readback.contents + row * pitch);
			switch (texture.pixelFormat) {
			case MTLPixelFormatB5G6R5Unorm: ConvertBGR565ToRGBA8888(dst, src, width); break;
			case MTLPixelFormatA1BGR5Unorm: ConvertABGR1555ToRGBA8888(dst, src, width); break;
			case MTLPixelFormatABGR4Unorm: ConvertABGR4444ToRGBA8888(dst, src, width); break;
			default: break;
			}
		}
	} else if (color) {
		if (texture.pixelFormat == MTLPixelFormatBGRA8Unorm) {
			Draw::ConvertFromBGRA8888((uint8_t *)pixels, (const uint8_t *)readback.contents, pixelStride, (uint32_t)pitch / 4, width, height, format);
		} else {
			Draw::ConvertFromRGBA8888((uint8_t *)pixels, (const uint8_t *)readback.contents, pixelStride, (uint32_t)pitch / 4, width, height, format);
		}
	} else if (format == DataFormat::D16) {
		Draw::ConvertToD16((uint8_t *)pixels, (const uint8_t *)readback.contents, pixelStride, (uint32_t)pitch / 4, width, height, DataFormat::D32F);
	} else {
		for (int row = 0; row < height; ++row) {
			memcpy((uint8_t *)pixels + row * pixelStride * bytesPerPixel, (uint8_t *)readback.contents + row * pitch, width * bytesPerPixel);
		}
	}
	return true;
}

}  // namespace Metal
