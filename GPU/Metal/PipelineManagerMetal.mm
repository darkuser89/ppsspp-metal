// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/PipelineManagerMetal.h"
#include "GPU/GPUDefinitions.h"

namespace {

struct VertexComponent {
	MTLVertexFormat format;
	uint32_t size;
};

// Decoder components are padded to 4 bytes (8 for short3). Match the existing
// Vulkan fetch conventions, including normalized packed colors and normals.
constexpr VertexComponent COMPONENTS[] = {
	{MTLVertexFormatInvalid, 0},
	{MTLVertexFormatFloat, 4},
	{MTLVertexFormatFloat2, 8},
	{MTLVertexFormatFloat3, 12},
	{MTLVertexFormatFloat4, 16},
	{MTLVertexFormatChar4Normalized, 4},
	{MTLVertexFormatShort4Normalized, 8},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUShort2Normalized, 4},
	{MTLVertexFormatUShort2Normalized, 4},
	{MTLVertexFormatUShort4Normalized, 8},
	{MTLVertexFormatUShort4Normalized, 8},
};
static_assert(ARRAY_SIZE(COMPONENTS) == DEC_U16_4 + 1);

bool SetAttribute(MTLVertexDescriptor *layout, PspAttributeLocation location, uint8_t component, uint32_t offset, uint32_t stride) {
	if (!component) {
		return true;
	}
	if (component >= ARRAY_SIZE(COMPONENTS) || offset > stride || COMPONENTS[component].size > stride - offset) {
		return false;
	}
	auto attr = layout.attributes[(int)location];
	attr.format = COMPONENTS[component].format;
	attr.offset = offset;
	attr.bufferIndex = Metal::VERTEX_BUFFER_SLOT;
	return true;
}

MTLVertexDescriptor *MakeLayout(const MetalGEVertexShader &shader, const DecVtxFormat *decoded, uint32_t *stride, std::string *error) {
	MTLVertexDescriptor *layout = [MTLVertexDescriptor vertexDescriptor];
	bool valid;
	if (shader.UseHWTransform()) {
		if (!decoded || !decoded->stride) {
			*error = "Metal hardware transform requires a decoded vertex layout";
			return nil;
		}
		*stride = decoded->stride;
		valid = SetAttribute(layout, PspAttributeLocation::POSITION, DEC_FLOAT_3, decoded->posoff, *stride) &&
			SetAttribute(layout, PspAttributeLocation::TEXCOORD, decoded->uvfmt, decoded->uvoff, *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR0, decoded->c0fmt, decoded->c0off, *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR1, decoded->c1fmt, decoded->c1off, *stride) &&
			SetAttribute(layout, PspAttributeLocation::NORMAL, decoded->nrmfmt, decoded->nrmoff, *stride) &&
			SetAttribute(layout, PspAttributeLocation::W1, decoded->w0fmt, decoded->w0off, *stride) &&
			SetAttribute(layout, PspAttributeLocation::W2, decoded->w1fmt, decoded->w1off, *stride);
	} else {
		*stride = sizeof(TransformedVertex);
		valid = SetAttribute(layout, PspAttributeLocation::POSITION, DEC_FLOAT_4, offsetof(TransformedVertex, pos), *stride) &&
			SetAttribute(layout, PspAttributeLocation::TEXCOORD, DEC_FLOAT_3, offsetof(TransformedVertex, uv), *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR0, DEC_U8_4, offsetof(TransformedVertex, color0_32), *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR1, DEC_U8_4, offsetof(TransformedVertex, color1_32), *stride) &&
			SetAttribute(layout, PspAttributeLocation::NORMAL, DEC_FLOAT_1, offsetof(TransformedVertex, fog), *stride);
	}
	if (!valid) {
		*error = "Metal decoded vertex component is invalid or extends beyond the stride";
		return nil;
	}
	for (MTLAttribute *attr in shader.function.stageInputAttributes) {
		if (attr.active && layout.attributes[attr.attributeIndex].format == MTLVertexFormatInvalid) {
			*error = "Metal vertex shader requires an attribute missing from the decoded layout";
			return nil;
		}
	}
	layout.layouts[Metal::VERTEX_BUFFER_SLOT].stride = *stride;
	layout.layouts[Metal::VERTEX_BUFFER_SLOT].stepFunction = MTLVertexStepFunctionPerVertex;
	return layout;
}

}

void PipelineManagerMetal::Clear() {
	pipelines_.clear();
	depthStates_.clear();
	shaderOwner_ = nullptr;
	shaderGeneration_ = 0;
}

void PipelineManagerMetal::DeviceLost() {
	Clear();
	manager_ = nullptr;
}

void PipelineManagerMetal::DeviceRestore(Metal::RenderManager *manager) {
	Clear();
	manager_ = manager;
}

const MetalGEPipeline *PipelineManagerMetal::GetOrCreate(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
	const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
	MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount) {
	error->clear();
	if (!manager_) {
		*error = "Metal pipeline manager has no rendering device";
		return nullptr;
	}
	if (sampleCount <= 0 || ![manager_->Context().Device() supportsTextureSampleCount:sampleCount]) {
		*error = "Unsupported Metal GE sample count";
		return nullptr;
	}
	// These are the attachment formats currently allocated by Metal Framebuffer.
	if ((colorFormat != MTLPixelFormatRGBA8Unorm && colorFormat != MTLPixelFormatBGRA8Unorm) ||
		(depthStencilFormat != MTLPixelFormatInvalid && depthStencilFormat != MTLPixelFormatDepth32Float_Stencil8)) {
		*error = "Unsupported Metal GE render target format";
		return nullptr;
	}
	const auto vertex = shaders.GetVertexShaderFromID(vertexID, error);
	if (!vertex) {
		return nullptr;
	}
	const auto fragment = shaders.GetFragmentShaderFromID(fragmentID, error);
	if (!fragment) {
		return nullptr;
	}
	if (shaderOwner_ != &shaders || shaderGeneration_ != shaders.CacheGeneration()) {
		pipelines_.clear();
		shaderOwner_ = &shaders;
		shaderGeneration_ = shaders.CacheGeneration();
	}
	if (vertex->UseHWTransform() && !decoded) {
		*error = "Metal hardware transform requires a decoded vertex layout";
		return nullptr;
	}
	PipelineKey key{vertexID.ToUint64(), fragmentID.ToUint64(), (uint64_t)colorFormat, (uint64_t)depthStencilFormat,
		blend.enabled, blend.srcColor, blend.dstColor, blend.colorOp, blend.srcAlpha, blend.dstAlpha, blend.alphaOp, blend.writeMask};
	key[20] = sampleCount;
	if (vertex->UseHWTransform()) {
		// The decoder ID normally implies offsets. Key the actual layout too, so
		// independently constructed layouts cannot silently reuse the wrong PSO.
		key[12] = decoded->stride;
		key[13] = decoded->posoff;
		key[14] = decoded->uvfmt | (decoded->uvoff << 8);
		key[15] = decoded->c0fmt | (decoded->c0off << 8);
		key[16] = decoded->c1fmt | (decoded->c1off << 8);
		key[17] = decoded->nrmfmt | (decoded->nrmoff << 8);
		key[18] = decoded->w0fmt | (decoded->w0off << 8);
		key[19] = decoded->w1fmt | (decoded->w1off << 8);
	}
	auto found = pipelines_.find(key);
	if (found != pipelines_.end()) {
		return &found->second;
	}
	MetalGEPipeline result;
	MTLVertexDescriptor *layout = MakeLayout(*vertex, decoded, &result.stride, error);
	if (!layout) {
		return nullptr;
	}
	MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
	desc.label = @"PSP GE pipeline";
	desc.vertexFunction = vertex->function;
	desc.fragmentFunction = fragment->function;
	desc.vertexDescriptor = layout;
	desc.depthAttachmentPixelFormat = depthStencilFormat;
	desc.stencilAttachmentPixelFormat = depthStencilFormat;
	desc.rasterSampleCount = sampleCount;
	auto color = desc.colorAttachments[0];
	color.pixelFormat = colorFormat;
	color.blendingEnabled = blend.enabled;
	color.sourceRGBBlendFactor = blend.srcColor;
	color.destinationRGBBlendFactor = blend.dstColor;
	color.rgbBlendOperation = blend.colorOp;
	color.sourceAlphaBlendFactor = blend.srcAlpha;
	color.destinationAlphaBlendFactor = blend.dstAlpha;
	color.alphaBlendOperation = blend.alphaOp;
	color.writeMask = blend.writeMask;
	NSError *nativeError = nil;
	result.state = [manager_->Context().Device() newRenderPipelineStateWithDescriptor:desc error:&nativeError];
	if (!result.state) {
		*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to create Metal GE pipeline";
		return nullptr;
	}
	for (const auto &resource : fragment->compiled.resources) {
		if (resource.kind == Metal::ResourceKind::SampledTexture) {
			result.textureMask |= 1u << resource.index;
		}
	}
	return &pipelines_.emplace(key, std::move(result)).first->second;
}

id<MTLDepthStencilState> PipelineManagerMetal::GetDepthStencil(const MetalGEDepthStencilState &state, std::string *error) {
	error->clear();
	if (!manager_) {
		*error = "Metal pipeline manager has no rendering device";
		return nil;
	}
	DepthKey key{state.depthWrite, state.depthCompare, state.stencilEnabled, state.stencilCompare,
		state.stencilFail, state.depthFail, state.pass, state.readMask, state.writeMask};
	auto found = depthStates_.find(key);
	if (found != depthStates_.end()) {
		return found->second;
	}
	MTLDepthStencilDescriptor *desc = [MTLDepthStencilDescriptor new];
	desc.label = @"PSP GE depth/stencil";
	desc.depthWriteEnabled = state.depthWrite;
	desc.depthCompareFunction = state.depthCompare;
	if (state.stencilEnabled) {
		MTLStencilDescriptor *stencil = [MTLStencilDescriptor new];
		stencil.stencilCompareFunction = state.stencilCompare;
		stencil.stencilFailureOperation = state.stencilFail;
		stencil.depthFailureOperation = state.depthFail;
		stencil.depthStencilPassOperation = state.pass;
		stencil.readMask = state.readMask;
		stencil.writeMask = state.writeMask;
		desc.frontFaceStencil = stencil;
		desc.backFaceStencil = stencil;
	}
	auto native = [manager_->Context().Device() newDepthStencilStateWithDescriptor:desc];
	if (!native) {
		*error = "Failed to create Metal GE depth/stencil state";
		return nil;
	}
	depthStates_.emplace(key, native);
	return native;
}
