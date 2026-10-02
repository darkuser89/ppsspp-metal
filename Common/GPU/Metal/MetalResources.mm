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
	case DataFormat::BC1_RGBA_UNORM_BLOCK:
	case DataFormat::BC2_UNORM_BLOCK:
	case DataFormat::BC3_UNORM_BLOCK:
	case DataFormat::BC4_UNORM_BLOCK:
	case DataFormat::BC5_UNORM_BLOCK:
	case DataFormat::BC7_UNORM_BLOCK:
		if (@available(macOS 13.0, iOS 16.4, *)) {
			switch (format) {
			case DataFormat::BC1_RGBA_UNORM_BLOCK: return MTLPixelFormatBC1_RGBA;
			case DataFormat::BC2_UNORM_BLOCK: return MTLPixelFormatBC2_RGBA;
			case DataFormat::BC3_UNORM_BLOCK: return MTLPixelFormatBC3_RGBA;
			case DataFormat::BC4_UNORM_BLOCK: return MTLPixelFormatBC4_RUnorm;
			case DataFormat::BC5_UNORM_BLOCK: return MTLPixelFormatBC5_RGUnorm;
			case DataFormat::BC7_UNORM_BLOCK: return MTLPixelFormatBC7_RGBAUnorm;
			default: break;
			}
		}
		return MTLPixelFormatInvalid;
	case DataFormat::ETC2_R8G8B8_UNORM_BLOCK:
	case DataFormat::ETC2_R8G8B8A1_UNORM_BLOCK:
	case DataFormat::ETC2_R8G8B8A8_UNORM_BLOCK:
	case DataFormat::ASTC_4x4_UNORM_BLOCK:
		if (@available(macOS 11.0, iOS 8.0, *)) {
			switch (format) {
			case DataFormat::ETC2_R8G8B8_UNORM_BLOCK: return MTLPixelFormatETC2_RGB8;
			case DataFormat::ETC2_R8G8B8A1_UNORM_BLOCK: return MTLPixelFormatETC2_RGB8A1;
			case DataFormat::ETC2_R8G8B8A8_UNORM_BLOCK: return MTLPixelFormatEAC_RGBA8;
			case DataFormat::ASTC_4x4_UNORM_BLOCK: return MTLPixelFormatASTC_4x4_LDR;
			default: break;
			}
		}
		return MTLPixelFormatInvalid;
	case DataFormat::D16:
		if (@available(macOS 13.0, iOS 16.0, *)) {
			return MTLPixelFormatDepth16Unorm;
		}
		return MTLPixelFormatInvalid;
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
	if (Draw::DataFormatIsBlockCompressed(format, nullptr)) {
		if (PixelFormat(format) == MTLPixelFormatInvalid) {
			return 0;
		}
		switch (format) {
		case Draw::DataFormat::BC1_RGBA_UNORM_BLOCK:
		case Draw::DataFormat::BC2_UNORM_BLOCK:
		case Draw::DataFormat::BC3_UNORM_BLOCK:
		case Draw::DataFormat::BC4_UNORM_BLOCK:
		case Draw::DataFormat::BC5_UNORM_BLOCK:
		case Draw::DataFormat::BC7_UNORM_BLOCK:
			if (@available(macOS 13.0, iOS 16.4, *)) {
				return device.supportsBCTextureCompression ? Draw::FMT_TEXTURE : 0;
			}
			return 0;
		default:
			if (@available(macOS 13.0, iOS 16.0, *)) {
				return [device supportsFamily:MTLGPUFamilyApple2] ? Draw::FMT_TEXTURE : 0;
			}
			return 0;
		}
	}
	if (format == Draw::DataFormat::R5G6B5_UNORM_PACK16 || format == Draw::DataFormat::R5G5B5A1_UNORM_PACK16 ||
		format == Draw::DataFormat::R4G4B4A4_UNORM_PACK16) {
		// Packed color formats are Apple GPU features, not part of the Mac2 baseline.
		if (@available(macOS 13.0, iOS 16.0, *)) {
			return [device supportsFamily:MTLGPUFamilyApple2] ? Draw::FMT_TEXTURE | Draw::FMT_RENDERTARGET | Draw::FMT_AUTOGEN_MIPS : 0;
		}
		return 0;
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

int MaxTextureDimension(id<MTLDevice> device) {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		if ([device supportsFamily:MTLGPUFamilyApple10]) {
			return 32768;
		}
	}
	return 16384;
}

bool SupportsDepthStencilResolve(id<MTLDevice> device) {
	return [device supportsFamily:MTLGPUFamilyApple5] || [device supportsFamily:MTLGPUFamilyMac2];
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
	const bool compressed = Draw::DataFormatIsBlockCompressed(desc.format, nullptr);
	const int maxDimension = volume ? 2048 : MaxTextureDimension(context.Device());
	if ((!volume && desc.type != Draw::TextureType::LINEAR2D) || (!volume && desc.depth != 1) ||
		(volume && compressed) || desc.depth <= 0 || desc.depth > maxDimension || desc.width <= 0 || desc.height <= 0 ||
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
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	if (storage) {
		if ((desc.format != Draw::DataFormat::R8G8B8A8_UNORM && desc.format != Draw::DataFormat::R16G16B16A16_FLOAT) || volume) {
			*error = "Metal compute uploads require a 2D RGBA8 or RGBA16F texture";
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
	std::vector<const uint8_t *> generatedData;
	const uint8_t *const *data = desc.initData.data();
	size_t dataSize = desc.initData.size();
	if (!dataSize && desc.initDataCallback) {
		generatedData.resize(desc.generateMips ? 1 : desc.mipLevels, nullptr);
		data = generatedData.data();
		dataSize = generatedData.size();
	}
	if ((dataSize && !texture->Upload(context, data, desc.initDataCallback, (int)dataSize, true, error)) ||
		(desc.generateMips && !dataSize)) {
		if (error->empty()) {
			*error = "Mipmap generation needs initialized base data";
		}
		texture->Release();
		return nullptr;
	}
	if (desc.generateMips && dataSize < (size_t)desc.mipLevels) {
		if (!EnsureCommands(context, error)) {
			texture->Release();
			return nullptr;
		}
		auto commands = context.InitializationCommands(error);
		if (!commands) {
			texture->Release();
			return nullptr;
		}
		// A view beginning at the last supplied mip preserves every explicit level.
		// Mipmap generation sees that level as its base and fills only the rest.
		id<MTLTexture> mipTarget = native;
		if (dataSize > 1) {
			const NSUInteger baseLevel = dataSize - 1;
			mipTarget = [native newTextureViewWithPixelFormat:native.pixelFormat textureType:native.textureType
				levels:NSMakeRange(baseLevel, desc.mipLevels - baseLevel) slices:NSMakeRange(0, 1)];
			if (!mipTarget) {
				*error = "Failed to create Metal mipmap generation view";
				texture->Release();
				return nullptr;
			}
		}
		id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
		if (!blit) {
			*error = "Failed to encode Metal mipmap generation";
			texture->Release();
			return nullptr;
		}
		[blit generateMipmapsForTexture:mipTarget];
		[blit endEncoding];
	}
	return texture;
}

bool Texture::Update(RenderContext &context, const uint8_t **data, Draw::TextureCallback callback, int levels, std::string *error) {
	return Upload(context, data, callback, levels, false, error);
}

bool Texture::UpdateRegions(RenderContext &context, int level, const Draw::TextureRegionUpdate *regions, int numRegions, std::string *error) {
	error->clear();
	if (level < 0 || (size_t)level >= texture_.mipmapLevelCount || !regions || numRegions <= 0 ||
		texture_.textureType != MTLTextureType2D || Draw::DataFormatIsBlockCompressed(format_, nullptr)) {
		*error = "Unsupported Metal texture region update";
		return false;
	}
	if (!EnsureCommands(context, error)) {
		return false;
	}
	const int width = std::max(1, width_ >> level);
	const int height = std::max(1, height_ >> level);
	const size_t pixelSize = Draw::DataFormatSizeInBytes(format_);
	struct PendingRegion {
		UploadSlice upload;
		size_t pitch;
	};
	std::vector<PendingRegion> pending;
	pending.reserve(numRegions);
	for (int i = 0; i < numRegions; ++i) {
		const auto &region = regions[i];
		if (region.x < 0 || region.y < 0 || region.w <= 0 || region.h <= 0 ||
			region.w > width || region.h > height || region.x > width - region.w || region.y > height - region.h ||
			!region.data || !pixelSize) {
			*error = "Invalid Metal texture update region";
			return false;
		}
		const size_t rowBytes = (size_t)region.w * pixelSize;
		if (region.byteStride < 0 || (region.byteStride && (size_t)region.byteStride < rowBytes)) {
			*error = "Invalid Metal texture update stride";
			return false;
		}
		const size_t srcStride = region.byteStride ? (size_t)region.byteStride : rowBytes;
		const size_t pitch = (rowBytes + 255) & ~size_t(255);
		std::vector<uint8_t> staging(pitch * region.h);
		for (int y = 0; y < region.h; ++y) {
			memcpy(staging.data() + pitch * y, region.data + srcStride * y, rowBytes);
		}
		UploadSlice upload = context.Upload(staging.data(), staging.size(), error);
		if (!upload) {
			return false;
		}
		pending.push_back({upload, pitch});
	}
	id<MTLBlitCommandEncoder> blit = [context.Commands() blitCommandEncoder];
	if (!blit) {
		*error = "Failed to start Metal texture update";
		return false;
	}
	for (int i = 0; i < numRegions; ++i) {
		const auto &region = regions[i];
		const auto &staged = pending[i];
		[blit copyFromBuffer:staged.upload.buffer sourceOffset:staged.upload.offset sourceBytesPerRow:staged.pitch
			sourceBytesPerImage:staged.pitch * region.h sourceSize:MTLSizeMake(region.w, region.h, 1)
			toTexture:texture_ destinationSlice:0 destinationLevel:level destinationOrigin:MTLOriginMake(region.x, region.y, 0)];
	}
	[blit endEncoding];
	return true;
}

id<MTLTexture> Texture::ArrayView() const {
	if (texture_.textureType != MTLTextureType2D) {
		return texture_;
	}
	if (!arrayView_) {
		arrayView_ = [texture_ newTextureViewWithPixelFormat:texture_.pixelFormat textureType:MTLTextureType2DArray
			levels:NSMakeRange(0, texture_.mipmapLevelCount) slices:NSMakeRange(0, 1)];
	}
	return arrayView_;
}

bool Texture::Upload(RenderContext &context, const uint8_t *const *data, Draw::TextureCallback callback, int levels,
	bool initialize, std::string *error) {
	error->clear();
	if (levels < 0 || (size_t)levels > texture_.mipmapLevelCount || (levels && !data)) {
		*error = "Invalid Metal texture upload levels";
		return false;
	}
	if (levels == 0) {
		return true;
	}
	if (!EnsureCommands(context, error)) {
		return false;
	}
	// Prepare the complete upload first, before changing the texture or opening an encoder.
	// The submission slot retains these bytes through both initialization and render commands.
	std::vector<UploadSlice> staging(levels);
	int blockSize = 0;
	const bool compressed = Draw::DataFormatIsBlockCompressed(format_, &blockSize);
	for (int level = 0; level < levels; ++level) {
		const int w = std::max(1, width_ >> level);
		const int h = std::max(1, height_ >> level);
		const int d = std::max(1, depth_ >> level);
		const size_t rowBytes = compressed ? (size_t)((w + 3) / 4) * blockSize : w * Draw::DataFormatSizeInBytes(format_);
		const size_t rows = compressed ? (h + 3) / 4 : h;
		const size_t pitch = (rowBytes + 255) & ~size_t(255);
		const size_t slicePitch = pitch * rows;
		if (@available(macOS 13.0, iOS 16.0, *)) {
			if (slicePitch * d > context.Device().maxBufferLength) {
				*error = "Metal texture upload exceeds the device buffer limit";
				return false;
			}
		} else {
			*error = "Metal 3 texture uploads require macOS 13 or iOS 16";
			return false;
		}
		UploadSlice upload = context.ReserveUpload(slicePitch * d, error);
		if (!upload) {
			return false;
		}
		uint8_t *contents = (uint8_t *)upload.buffer.contents + upload.offset;
		bool generated = callback && callback(contents, data[level], w, h, d, (uint32_t)pitch, (uint32_t)slicePitch);
		// Thin3d callbacks return false to request a copy from the original data.
		if (!generated) {
			if (!data[level]) {
				*error = "Missing Metal texture mip data";
				return false;
			}
			for (int z = 0; z < d; ++z) {
				for (size_t y = 0; y < rows; ++y) {
					memcpy(contents + slicePitch * z + pitch * y, data[level] + rowBytes * (rows * z + y), rowBytes);
				}
			}
		}
		staging[level] = upload;
	}
	id<MTLBlitCommandEncoder> blit = initialize ? context.InitializationBlitEncoder(error) : [context.Commands() blitCommandEncoder];
	if (!blit) {
		if (error->empty()) {
			*error = "Failed to encode Metal texture upload";
		}
		return false;
	}
	for (int level = 0; level < levels; ++level) {
		const int w = std::max(1, width_ >> level);
		const int h = std::max(1, height_ >> level);
		const size_t rowBytes = compressed ? (size_t)((w + 3) / 4) * blockSize : w * Draw::DataFormatSizeInBytes(format_);
		const size_t pitch = (rowBytes + 255) & ~size_t(255);
		const size_t rows = compressed ? (h + 3) / 4 : h;
		[blit copyFromBuffer:staging[level].buffer sourceOffset:staging[level].offset sourceBytesPerRow:pitch sourceBytesPerImage:pitch * rows
			sourceSize:MTLSizeMake(w, h, std::max(1, depth_ >> level)) toTexture:texture_ destinationSlice:0 destinationLevel:level destinationOrigin:MTLOriginMake(0, 0, 0)];
	}
	if (!initialize) {
		[blit endEncoding];
	}
	return true;
}

Framebuffer *Framebuffer::Create(id<MTLDevice> device, const Draw::FramebufferDesc &desc, std::string *error) {
	error->clear();
	const int maxDimension = MaxTextureDimension(device);
	if (desc.width <= 0 || desc.height <= 0 || desc.width > maxDimension || desc.height > maxDimension ||
		desc.numLayers < 1 || desc.numLayers > 2 || desc.multiSampleLevel < 0 || desc.multiSampleLevel > 4 ||
		![device supportsTextureSampleCount:1u << desc.multiSampleLevel]) {
		*error = "Unsupported Metal framebuffer dimensions, layers, or sample count";
		return nullptr;
	}
	if (desc.multiSampleLevel > 0 && desc.z_stencil) {
		if (!SupportsDepthStencilResolve(device)) {
			*error = "Metal multisample depth/stencil resolve is unavailable";
			return nullptr;
		}
	}
	if (desc.numLayers > 1 && desc.multiSampleLevel > 0) {
		bool layeredMSAA = false;
		if (@available(macOS 13.0, iOS 16.0, *)) {
			layeredMSAA = [device supportsFamily:MTLGPUFamilyApple7] || [device supportsFamily:MTLGPUFamilyMac2];
		}
		if (!layeredMSAA) {
			*error = "Metal multisample array framebuffers require Apple7 or Mac2";
			return nullptr;
		}
	}
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:desc.width height:desc.height mipmapped:NO];
	td.textureType = desc.numLayers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
	td.arrayLength = desc.numLayers;
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
		if (desc.numLayers > 1) {
			if (@available(macOS 13.0, iOS 16.0, *)) {
				td.textureType = MTLTextureType2DMultisampleArray;
			}
		} else {
			td.textureType = MTLTextureType2DMultisample;
		}
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
	fbo->layers_ = desc.numLayers;
	fbo->multiSampleLevel_ = desc.multiSampleLevel;
	if (desc.numLayers > 1) {
		auto layerView = [](id<MTLTexture> texture, int layer) -> id<MTLTexture> {
			return texture ? [texture newTextureViewWithPixelFormat:texture.pixelFormat
				textureType:MTLTextureType2D
				levels:NSMakeRange(0, 1) slices:NSMakeRange(layer, 1)] : nil;
		};
		for (int layer = 0; layer < desc.numLayers; ++layer) {
			fbo->colorLayers_[layer] = layerView(color, layer);
			fbo->depthStencilLayers_[layer] = layerView(depthStencil, layer);
			if (!fbo->colorLayers_[layer] || (desc.z_stencil && !fbo->depthStencilLayers_[layer])) {
				*error = "Failed to create Metal framebuffer layer views";
				delete fbo;
				return nullptr;
			}
		}
	}
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

id<MTLTexture> Framebuffer::ColorArray() const {
	if (layers_ > 1) {
		return color_;
	}
	if (!colorArray_ && color_) {
		colorArray_ = [color_ newTextureViewWithPixelFormat:color_.pixelFormat
			textureType:MTLTextureType2DArray levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
	}
	return colorArray_;
}

id<MTLTexture> Framebuffer::DepthStencilArray() const {
	if (layers_ > 1) {
		return depthStencil_;
	}
	if (!depthStencilArray_ && depthStencil_) {
		depthStencilArray_ = [depthStencil_ newTextureViewWithPixelFormat:depthStencil_.pixelFormat
			textureType:MTLTextureType2DArray levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
	}
	return depthStencilArray_;
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

static id<MTLTexture> RootTexture(id<MTLTexture> texture) {
	while (texture.parentTexture) {
		texture = texture.parentTexture;
	}
	return texture;
}

bool Framebuffer::OwnsTexture(id<MTLTexture> texture) const {
	for (id<MTLTexture> ancestor = texture; ancestor; ancestor = ancestor.parentTexture) {
		if (ancestor == color_ || ancestor == depthStencil_ || ancestor == multisampleColor_ || ancestor == multisampleDepthStencil_) {
			return true;
		}
	}
	return false;
}

void Framebuffer::SetRenderAttachments(MTLRenderPassDescriptor *pass, int layer) const {
	if (layer < 0 && layers_ > 1) {
		pass.renderTargetArrayLength = layers_;
	}
	pass.colorAttachments[0].texture = layer < 0 ? ColorAttachment() : ColorAttachmentLayer(layer);
	if (layer >= 0 && multisampleColor_ && layers_ > 1) {
		pass.colorAttachments[0].slice = layer;
	}
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	if (multisampleColor_) {
		pass.colorAttachments[0].resolveTexture = layer < 0 || layers_ > 1 ? color_ : ColorLayer(layer);
		if (layer >= 0 && layers_ > 1) {
			pass.colorAttachments[0].resolveSlice = layer;
		}
		pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
	}
	if (depthStencil_) {
		pass.depthAttachment.texture = layer < 0 ? DepthStencilAttachment() : DepthStencilAttachmentLayer(layer);
		pass.stencilAttachment.texture = pass.depthAttachment.texture;
		if (layer >= 0 && multisampleDepthStencil_ && layers_ > 1) {
			pass.depthAttachment.slice = layer;
			pass.stencilAttachment.slice = layer;
		}
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.stencilAttachment.storeAction = MTLStoreActionStore;
		if (multisampleDepthStencil_) {
			pass.depthAttachment.resolveTexture = layer < 0 || layers_ > 1 ? depthStencil_ : DepthStencilLayer(layer);
			pass.stencilAttachment.resolveTexture = pass.depthAttachment.resolveTexture;
			if (layer >= 0 && layers_ > 1) {
				pass.depthAttachment.resolveSlice = layer;
				pass.stencilAttachment.resolveSlice = layer;
			}
			pass.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
			pass.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
			pass.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
			pass.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
		}
	}
}

static bool ValidRect(id<MTLTexture> texture, int x, int y, int w, int h) {
	bool supportedType = texture.textureType == MTLTextureType2D || texture.textureType == MTLTextureType2DArray ||
		texture.textureType == MTLTextureType2DMultisample;
	if (@available(macOS 13.0, iOS 16.0, *)) {
		supportedType |= texture.textureType == MTLTextureType2DMultisampleArray;
	}
	return texture && supportedType &&
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
	// Metal permits disjoint copies within one texture. Overlapping regions
	// need a snapshot so the source stays unchanged throughout the copy.
	id<MTLTexture> temporary = nil;
	const bool overlap = srcX < dstX + width && dstX < srcX + width &&
		srcY < dstY + height && dstY < srcY + height;
	if (overlap && RootTexture(src) == RootTexture(dst)) {
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:width height:height mipmapped:NO];
		td.textureType = src.textureType;
		td.arrayLength = src.arrayLength;
		td.sampleCount = src.sampleCount;
		td.storageMode = MTLStorageModePrivate;
		temporary = [context.Device() newTextureWithDescriptor:td];
		if (!temporary) {
			*error = "Failed to allocate Metal copy snapshot";
			return false;
		}
	}
	id<MTLBlitCommandEncoder> blit = [context.Commands() blitCommandEncoder];
	if (!blit) {
		*error = "Failed to encode Metal image copy";
		return false;
	}
	const NSUInteger layers = std::min(src.arrayLength, dst.arrayLength);
	if (temporary) {
		for (NSUInteger layer = 0; layer < layers; ++layer) {
			[blit copyFromTexture:src sourceSlice:layer sourceLevel:0 sourceOrigin:MTLOriginMake(srcX, srcY, 0)
				sourceSize:MTLSizeMake(width, height, 1) toTexture:temporary destinationSlice:layer destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
		}
		src = temporary;
		srcX = srcY = 0;
	}
	for (NSUInteger layer = 0; layer < layers; ++layer) {
		[blit copyFromTexture:src sourceSlice:layer sourceLevel:0 sourceOrigin:MTLOriginMake(srcX, srcY, 0)
			sourceSize:MTLSizeMake(width, height, 1) toTexture:dst destinationSlice:layer destinationLevel:0 destinationOrigin:MTLOriginMake(dstX, dstY, 0)];
	}
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
	if (!blit) {
		*error = "Failed to encode Metal depth/stencil copy";
		return false;
	}
	for (NSUInteger layer = 0; layer < std::min(src.arrayLength, dst.arrayLength); ++layer) {
		[blit copyFromTexture:src sourceSlice:layer sourceLevel:0 sourceOrigin:MTLOriginMake(srcX, srcY, 0)
			sourceSize:MTLSizeMake(width, height, 1) toBuffer:plane destinationOffset:0
			destinationBytesPerRow:pitch destinationBytesPerImage:pitch * height options:option];
		[blit copyFromBuffer:plane sourceOffset:0 sourceBytesPerRow:pitch sourceBytesPerImage:pitch * height
			sourceSize:MTLSizeMake(width, height, 1) toTexture:dst destinationSlice:layer destinationLevel:0
			destinationOrigin:MTLOriginMake(dstX, dstY, 0) options:option];
	}
	[blit endEncoding];
	return true;
}

bool FramebufferCopy::Copy(RenderContext &context, Framebuffer *src, int sx, int sy, Framebuffer *dst,
	int dx, int dy, int width, int height, Draw::Aspect aspects, std::string *error) {
	return Blit(context, src, sx, sy, width, height, dst, dx, dy, width, height, aspects, Draw::FB_BLIT_NEAREST, error);
}

bool FramebufferCopy::Blit(RenderContext &context, Framebuffer *src, int sx, int sy, int srcWidth, int srcHeight,
	Framebuffer *dst, int dx, int dy, int dstWidth, int dstHeight, Draw::Aspect aspects,
	Draw::FBBlitFilter filter, std::string *error) {
	error->clear();
	const bool color = aspects & Draw::Aspect::COLOR_BIT;
	const bool depth = aspects & Draw::Aspect::DEPTH_BIT;
	const bool stencil = aspects & Draw::Aspect::STENCIL_BIT;
	const auto allowed = Draw::Aspect::COLOR_BIT | Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT;
	const bool linear = filter == Draw::FB_BLIT_LINEAR;
	if (!src || !dst || !(color || depth || stencil) || (aspects & ~allowed) ||
		!ValidRect(src->Color(), sx, sy, srcWidth, srcHeight) || !ValidRect(dst->Color(), dx, dy, dstWidth, dstHeight) ||
		(linear && (!color || depth || stencil || src->SampleCount() > 1)) ||
		((depth || stencil) && (!src->DepthStencil() || !dst->DepthStencil()))) {
		*error = "Invalid Metal framebuffer transfer region or aspects";
		return false;
	}
	// Read from a separate resource even for non-overlapping self-copies: Metal
	// forbids sampling an attachment in its own render pass.
	Draw::AutoRef<Framebuffer> snapshot;
	if (src == dst) {
		snapshot.reset(Framebuffer::Create(context.Device(), {srcWidth, srcHeight, 1, src->Layers(), src->MultiSampleLevel(), depth || stencil, "Metal transfer snapshot"}, error));
		if (!snapshot ||
			(color && !CopyImage(context, src->ColorAttachment(), sx, sy, snapshot->ColorAttachment(), 0, 0, srcWidth, srcHeight, error)) ||
			((depth || stencil) && !CopyImage(context, src->DepthStencilAttachment(), sx, sy, snapshot->DepthStencilAttachment(), 0, 0, srcWidth, srcHeight, error))) {
			return false;
		}
		src = snapshot.ptr;
		sx = sy = 0;
	}
	const bool perSample = src->SampleCount() > 1 && src->SampleCount() == dst->SampleCount();
	const bool arraySamples = perSample && src->Layers() > 1;
	const auto depthFormat = dst->DepthStencil() ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid;
	const std::array<uint64_t, 7> key{(uint64_t)aspects, perSample, arraySamples, linear, (uint64_t)dst->SampleCount(), (uint64_t)dst->Color().pixelFormat, (uint64_t)depthFormat};
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
		shader.source += "};\nfragment Out copyFragment(float4 pos [[position]], constant float4 &region [[buffer(0)]], constant float2 &dstOrigin [[buffer(1)]], constant float2 &invSize [[buffer(2)]]";
		const std::string textureType = arraySamples ? "texture2d_ms_array" : perSample ? "texture2d_ms" : "texture2d";
		if (perSample) {
			shader.source += ", uint sample [[sample_id]]";
		}
		if (arraySamples) {
			shader.source += ", constant uint &sourceLayer [[buffer(3)]]";
		}
		if (color) {
			shader.source += ", " + textureType + (linear ? "<float, access::sample> colorTex [[texture(0)]]" : "<float, access::read> colorTex [[texture(0)]]");
		}
		if (depth) {
			shader.source += ", " + textureType + "<float, access::read> depthTex [[texture(1)]]";
		}
		if (stencil) {
			shader.source += ", " + textureType + "<uint, access::read> stencilTex [[texture(2)]]";
		}
		shader.source += ") { float2 sourcePos = (pos.xy - dstOrigin) * region.zw + region.xy; uint2 p = uint2(sourcePos); Out out;\n";
		const std::string read = arraySamples ? ".read(p, sourceLayer, sample)" : perSample ? ".read(p, sample)" : ".read(p)";
		if (color) {
			if (linear) {
				shader.source += "constexpr sampler linearSampler(filter::linear, address::clamp_to_edge); out.color = colorTex.sample(linearSampler, sourcePos * invSize);\n";
			} else {
				shader.source += "out.color = colorTex" + read + ";\n";
			}
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
	const float region[4] = {(float)sx, (float)sy, (float)srcWidth / dstWidth, (float)srcHeight / dstHeight};
	const float dstOrigin[2] = {(float)dx, (float)dy};
	for (int layer = 0; layer < std::min(src->Layers(), dst->Layers()); ++layer) {
		id<MTLTexture> srcColor = perSample ? src->ColorAttachmentLayer(layer) : src->ColorLayer(layer);
		id<MTLTexture> srcDepth = perSample ? src->DepthStencilAttachmentLayer(layer) : src->DepthStencilLayer(layer);
		id<MTLTexture> srcStencil = stencil ? [srcDepth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8] : nil;
		if (stencil && !srcStencil) {
			*error = "Failed to create Metal stencil transfer view";
			return false;
		}
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		dst->SetRenderAttachments(rp, layer);
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
		[encoder setViewport:(MTLViewport){(double)dx, (double)dy, (double)dstWidth, (double)dstHeight, 0, 1}];
		[encoder setScissorRect:(MTLScissorRect){(NSUInteger)dx, (NSUInteger)dy, (NSUInteger)dstWidth, (NSUInteger)dstHeight}];
		const float invSize[2] = {1.0f / srcColor.width, 1.0f / srcColor.height};
		[encoder setFragmentBytes:region length:sizeof(region) atIndex:0];
		[encoder setFragmentBytes:dstOrigin length:sizeof(dstOrigin) atIndex:1];
		[encoder setFragmentBytes:invSize length:sizeof(invSize) atIndex:2];
		if (arraySamples) {
			const uint32_t sourceLayer = layer;
			[encoder setFragmentBytes:&sourceLayer length:sizeof(sourceLayer) atIndex:3];
		}
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
	}
	return true;
}

bool Readback(RenderContext &context, id<MTLTexture> texture, Draw::Aspect aspect,
	int x, int y, int width, int height, Draw::DataFormat format, void *pixels, int pixelStride, std::string *error, int level, int z) {
	using Draw::DataFormat;
	error->clear();
	// PSP framebuffers can have overlapping rows, including stride zero. The
	// shared converters write rows in order, matching the other backends.
	if (!pixels || pixelStride < 0 || !texture || level < 0 || (NSUInteger)level >= texture.mipmapLevelCount ||
		(texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray && texture.textureType != MTLTextureType3D) ||
		z < 0 || (NSUInteger)z >= (texture.textureType == MTLTextureType3D ? std::max<NSUInteger>(1, texture.depth >> level) : texture.arrayLength) ||
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
	// FramebufferManagerCommon::ReadbackStencilbuffer requests S8 with DEPTH_BIT.
	const bool stencil = format == DataFormat::S8 &&
		(aspect == Draw::Aspect::STENCIL_BIT || aspect == Draw::Aspect::DEPTH_BIT);
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
			(format != DataFormat::R8G8B8A8_UNORM && format != DataFormat::B8G8R8A8_UNORM && format != DataFormat::R8G8B8_UNORM &&
			format != DataFormat::R5G6B5_UNORM_PACK16 && format != DataFormat::A1R5G5B5_UNORM_PACK16 && format != DataFormat::A4R4G4B4_UNORM_PACK16)) {
			*error = "Unsupported Metal color readback format";
			return false;
		}
	} else if (texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
		if (stencil) {
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
	if (!blit) {
		*error = "Failed to encode Metal readback";
		return false;
	}
	[blit copyFromTexture:texture sourceSlice:texture.textureType == MTLTextureType2DArray ? z : 0
		sourceLevel:level sourceOrigin:MTLOriginMake(x, y, texture.textureType == MTLTextureType3D ? z : 0)
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
		} else if (format == DataFormat::B8G8R8A8_UNORM) {
			for (int row = 0; row < height; ++row) {
				auto dst = (uint32_t *)pixels + row * pixelStride;
				auto src = (const uint32_t *)((const uint8_t *)readback.contents + row * pitch);
				ConvertRGBA8888ToBGRA8888(dst, src, width);
			}
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
