// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/DrawEngineMetal.h"
#include "GPU/Metal/FramebufferManagerMetal.h"
#include "GPU/Common/SoftwareTransformCommon.h"
#include "Common/Data/Convert/SmallDataConvert.h"

DrawEngineMetal::DrawEngineMetal(Draw::DrawContext *draw) {
	DeviceRestore(draw);
}

DrawEngineMetal::~DrawEngineMetal() {
	DeviceLost();
}

void DrawEngineMetal::DeviceLost() {
	if (draw_) {
		draw_->SetInvalidationCallback({});
	}
	pipelines_.DeviceLost();
	samplers_.Clear();
	manager_ = nullptr;
	draw_ = nullptr;
}

void DrawEngineMetal::DeviceRestore(Draw::DrawContext *draw) {
	DeviceLost();
	draw_ = draw;
	manager_ = draw ? (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER) : nullptr;
	pipelines_.DeviceRestore(manager_);
	if (draw_) {
		draw_->SetInvalidationCallback([this](InvalidationCallbackFlags flags) { Invalidate(flags); });
	}
	gstate_c.Dirty(DIRTY_ALL_RENDER_STATE | DIRTY_ALL_UNIFORMS);
}

void DrawEngineMetal::NotifyConfigChanged() {
	DrawEngineCommon::NotifyConfigChanged();
	pipelines_.Clear();
	samplers_.Clear();
}

void DrawEngineMetal::BeginFrame() {
	DrawEngineCommon::BeginFrame();
	gstate_c.Dirty(DIRTY_ALL_RENDER_STATE);
}

void DrawEngineMetal::Invalidate(InvalidationCallbackFlags flags) {
	// GE native state is always rebound. Thin3d may also have replaced texture bindings.
	gstate_c.Dirty(DIRTY_ALL_RENDER_STATE | DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS);
}

bool DrawEngineMetal::ApplyDrawState(GEPrimitiveType prim, MetalDrawState *state, ViewportAndScissor *viewport, std::string *error) {
	pipelineState_ = {};
	if (!gstate.isModeClear()) {
		pipelineState_.Convert(draw_->GetShaderLanguageDesc().bitwiseOps, gstate_c.Use(GPU_USE_SHADER_BLENDING));
		if (pipelineState_.FramebufferRead()) {
			FBOTexState binding = FBO_TEX_NONE;
			ApplyFramebufferRead(&binding);
			if (binding != FBO_TEX_COPY_BIND_TEX || !framebufferManager_->GetCurrentRenderVFB()) {
				*error = "Metal shader blending requires a copyable current framebuffer";
				return false;
			}
			auto &blend = pipelineState_.blendState;
			ApplyStencilReplaceAndLogicOpIgnoreBlend(blend.replaceAlphaWithStencil, blend);
			if (!framebufferManager_->BindFramebufferAsColorTexture(DRAW_BINDING_2ND_TEXTURE,
				framebufferManager_->GetCurrentRenderVFB(), BINDFBCOLOR_MAY_COPY | BINDFBCOLOR_UNCACHED, 0)) {
				*error = "Metal shader blending could not bind its framebuffer copy";
				return false;
			}
			SamplerCacheKey key{};
			key.sClamp = true;
			key.tClamp = true;
			manager_->SetNativeSampler(DRAW_BINDING_2ND_TEXTURE, samplers_.GetOrCreate(manager_->Context().Device(), key, error));
			if (!error->empty()) {
				return false;
			}
			framebufferManager_->RebindFramebuffer("Metal shader blending");
			gstate_c.Dirty(DIRTY_FRAGMENTSHADER_STATE);
		}
		if (pipelineState_.blendState.dirtyShaderBlendFixValues) {
			gstate_c.Dirty(DIRTY_SHADERBLEND);
		}
	}
	if (!ConvertMetalDrawState(prim, pipelineState_, state, error)) {
		return false;
	}
	if (!gstate.isModeClear() && !IsDepthTestEffectivelyDisabled()) {
		UpdateEverUsedEqualDepth(gstate.getDepthTestFunction());
	}
	ConvertViewportAndScissor(framebufferManager_->GetDisplayLayoutConfigCopy(), framebufferManager_->UseBufferedRendering(),
		framebufferManager_->GetRenderWidth(), framebufferManager_->GetRenderHeight(),
		framebufferManager_->GetTargetBufferWidth(), framebufferManager_->GetTargetBufferHeight(), *viewport);
	return true;
}

bool DrawEngineMetal::FlushDraw(std::string *error) {
	if (!manager_ || !shaderManager_ || !textureCache_ || !framebufferManager_ || !dec_) {
		*error = "Metal draw engine is missing its device or GE managers";
		return false;
	}
	GEPrimitiveType prim = prevPrim_;
	bool hardware = CanUseHardwareTransform(prim) && gstate.getShadeMode() != GE_SHADE_FLAT;
	if ((clipInfoFlags_ & ClipInfoFlags::Valid) && (clipInfoFlags_ & ClipInfoFlags::SoftClipCull)) {
		hardware = false;
	}
	if (clipInfoFlags_ != lastClipInfoFlags_ || hardware != lastUseHwTransform_) {
		gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_RASTER_STATE | DIRTY_TEXTURE_PARAMS);
		lastClipInfoFlags_ = clipInfoFlags_;
		lastUseHwTransform_ = hardware;
	}
	DecodeVerts(dec_, decoded_);
	int count, maxIndex;
	bool indexed;
	DecodeIndsAndGetData(&prim, &count, &maxIndex, &indexed, !hardware);
	const bool hasColor = (lastVType_ & GE_VTYPE_COL_MASK) != GE_VTYPE_COL_NONE;
	if (gstate.isModeThrough()) {
		gstate_c.vertexFullAlpha &= hasColor || gstate.getMaterialAmbientA() == 255;
	} else {
		gstate_c.vertexFullAlpha &= ((hasColor && (gstate.materialupdate & 1)) || gstate.getMaterialAmbientA() == 255) &&
			(!gstate.isLightingEnabled() || gstate.getAmbientA() == 255);
	}
	TextureApplyResult texture;
	bool applyTexture = !gstate.isModeClear() && gstate.isTextureMapEnabled();
	if (applyTexture) {
		gstate_c.Clean(DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS);
		gstate_c.dstSquared = false;
		texture = textureCache_->ApplyTexture(true);
	}
	uint16_t *indices = decIndex_;
	const void *vertices = decoded_;
	int vertexCount = numDecodedVerts_;
	SoftwareTransformResult transformed{};
	SoftwareTransformAction action = SW_DRAW_INDEXED;
	if (!hardware) {
		if (gstate.getShadeMode() == GE_SHADE_FLAT) {
			IndexBufferProvokingLastToFirst(prim, indices, count);
		}
		if (useDepthRaster_) {
			DepthRasterPredecoded(prim, decoded_, numDecodedVerts_, dec_, count);
		}
		SoftwareTransformParams params{};
		params.decoded = decoded_;
		params.transformed = transformed_;
		params.transformedExpanded = transformedExpanded_;
		params.everUsedEqualDepth = everUsedEqualDepth_;
		params.clipInfoFlags = clipInfoFlags_;
		// Metal load-action clears cover the entire attachment. Partial rectangles
		// and separate color/alpha masks go through the shared raster clear path.
		params.allowClear = framebufferManager_->UseBufferedRendering() &&
			gstate.getScissorX1() == 0 && gstate.getScissorY1() == 0 &&
			gstate.getScissorX2() + 1 >= framebufferManager_->GetTargetBufferWidth() &&
			gstate.getScissorY2() + 1 >= framebufferManager_->GetTargetBufferHeight();
		params.allowSeparateAlphaClear = false;
		action = RunSoftwareTransform(params, prim, dec_->VertexType(), dec_->GetDecVtxFmt(), numDecodedVerts_,
			VERTEX_BUFFER_MAX, count, indices, RemainingIndices(indices), &transformed);
		vertices = transformed.drawBuffer;
		vertexCount = transformed.drawVertexCount;
		count = transformed.drawIndexCount;
		indexed = true;
	}
	if (applyTexture) {
		textureCache_->ApplySampler(texture, clipInfoFlags_ & ClipInfoFlags::FlatZ, transformed.pixelMapped);
	}
	if (action == SW_CULLED) {
		return true;
	}
	// Shared texture/depth conversions can leave an intermediate target bound.
	// Restore the GE target before choosing attachment formats or drawing.
	auto *vfb = framebufferManager_->GetCurrentRenderVFB();
	if (vfb && vfb->fbo && manager_->RenderTarget() != vfb->fbo) {
		framebufferManager_->RebindFramebuffer("Metal GE target restore");
	}
	MetalDrawState state;
	ViewportAndScissor viewport;
	if (!ApplyDrawState(prim, &state, &viewport, error)) {
		return false;
	}
	if (action == SW_CLEAR) {
		Draw::Aspect aspects = Draw::Aspect::NO_BIT;
		if (gstate.isClearModeColorMask()) {
			aspects |= Draw::Aspect::COLOR_BIT;
		}
		if (gstate.isClearModeAlphaMask()) {
			aspects |= Draw::Aspect::STENCIL_BIT;
		}
		if (gstate.isClearModeDepthMask()) {
			aspects |= Draw::Aspect::DEPTH_BIT;
		}
		draw_->Clear(aspects, transformed.color, transformed.depth, transformed.color >> 24);
		if (gstate_c.Use(GPU_USE_CLEAR_RAM_HACK) && gstate.isClearModeColorMask() &&
			(gstate.isClearModeAlphaMask() || gstate_c.framebufFormat == GE_FORMAT_565)) {
			framebufferManager_->ApplyClearToMemory(gstate.getScissorX1(), gstate.getScissorY1(),
				gstate.getScissorX2() + 1, gstate.getScissorY2() + 1, transformed.color);
		}
		return true;
	}
	if (count <= 0 || vertexCount <= 0) {
		return true;
	}
	const MetalGEVertexShader *vs;
	const MetalGEFragmentShader *fs;
	if (!shaderManager_->GetShaders(dec_->VertexType(), pipelineState_, hardware, clipInfoFlags_, &vs, &fs, error)) {
		return false;
	}
	const auto *pipeline = pipelines_.GetOrCreate(*shaderManager_, vs->id, fs->id, hardware ? &dec_->GetDecVtxFmt() : nullptr,
		state.blend, manager_->ColorFormat(), manager_->DepthStencilFormat(), error, manager_->SampleCount());
	if (!pipeline) {
		return false;
	}
	auto depth = pipelines_.GetDepthStencil(state.depthStencil, error);
	if (!depth) {
		return false;
	}
	auto encoder = manager_->RenderEncoder();
	if (!encoder) {
		*error = "Metal could not open the GE render pass";
		return false;
	}
	if (!shaderManager_->UpdateUniforms(framebufferManager_->UseBufferedRendering(), transformed.pixelMapped, error)) {
		return false;
	}
	auto &context = manager_->Context();
	auto vb = context.Upload(vertices, (size_t)vertexCount * pipeline->stride, error);
	if (!vb) {
		return false;
	}
	Metal::UploadSlice ib;
	if (indexed) {
		ib = context.Upload(indices, (size_t)count * sizeof(uint16_t), error);
	}
	if (indexed && !ib) {
		return false;
	}
	int x = std::max(0, viewport.scissorX);
	int y = std::max(0, viewport.scissorY);
	int w = std::min(framebufferManager_->GetRenderWidth(), viewport.scissorX + std::max(0, viewport.scissorW)) - x;
	int h = std::min(framebufferManager_->GetRenderHeight(), viewport.scissorY + std::max(0, viewport.scissorH)) - y;
	if (w <= 0 || h <= 0 || viewport.viewportW <= 0 || viewport.viewportH <= 0) {
		return true;
	}
	if (!shaderManager_->BindUniforms(encoder, error) || !manager_->BindTextures(encoder, pipeline->textureMask, error)) {
		return false;
	}
	[encoder setRenderPipelineState:pipeline->state];
	[encoder setDepthStencilState:depth];
	[encoder setStencilReferenceValue:transformed.setStencil ? transformed.stencilValue : state.stencilRef];
	[encoder setCullMode:state.cull];
	[encoder setFrontFacingWinding:MTLWindingCounterClockwise];
	[encoder setDepthClipMode:state.depthClip];
	[encoder setDepthBias:0 slopeScale:0 clamp:0];
	[encoder setViewport:(MTLViewport){viewport.viewportX, viewport.viewportY, viewport.viewportW, viewport.viewportH, 0, 1}];
	[encoder setScissorRect:(MTLScissorRect){(NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h}];
	float blendColor[4];
	Uint8x4ToFloat4(blendColor, state.blendColor);
	[encoder setBlendColorRed:blendColor[0] green:blendColor[1] blue:blendColor[2] alpha:blendColor[3]];
	[encoder setVertexBuffer:vb.buffer offset:vb.offset atIndex:Metal::VERTEX_BUFFER_SLOT];
	MTLPrimitiveType topology = hardware && prim == GE_PRIM_TRIANGLE_STRIP ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle;
	if (indexed) {
		[encoder drawIndexedPrimitives:topology indexCount:count indexType:MTLIndexTypeUInt16 indexBuffer:ib.buffer indexBufferOffset:ib.offset];
	} else {
		[encoder drawPrimitives:topology vertexStart:0 vertexCount:count];
	}
	if (hardware && useDepthRaster_) {
		DepthRasterSubmitRaw(prim, dec_, dec_->VertexType(), count);
	}
	gpuStats.perFrame.numVertsDrawn += count;
	return true;
}

void DrawEngineMetal::Flush() {
	if (!numDrawVerts_) {
		return;
	}
	lastError_.clear();
	if (!FlushDraw(&lastError_)) {
		ERROR_LOG(Log::G3D, "Metal draw failed: %s", lastError_.c_str());
		gstate_c.Dirty(DIRTY_ALL_RENDER_STATE | DIRTY_ALL_UNIFORMS);
	}
	ResetAfterDrawInline();
	if (framebufferManager_) {
		framebufferManager_->SetColorUpdated(gstate_c.skipDrawReason);
	}
	if (gpuCommon_) {
		gpuCommon_->NotifyFlush();
	}
}
