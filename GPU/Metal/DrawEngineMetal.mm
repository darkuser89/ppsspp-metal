// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/DrawEngineMetal.h"
#include "GPU/Metal/FramebufferManagerMetal.h"
#include "GPU/Common/SoftwareTransformCommon.h"
#include "Common/Data/Convert/SmallDataConvert.h"

namespace {
bool SameViewport(const MTLViewport &a, const MTLViewport &b) {
	return a.originX == b.originX && a.originY == b.originY && a.width == b.width && a.height == b.height &&
		a.znear == b.znear && a.zfar == b.zfar;
}

bool SameScissor(const MTLScissorRect &a, const MTLScissorRect &b) {
	return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}
}

DrawEngineMetal::DrawEngineMetal(Draw::DrawContext *draw) {
	DeviceRestore(draw);
}

DrawEngineMetal::~DrawEngineMetal() {
	DeviceLost();
}

void DrawEngineMetal::DeviceLost() {
	encoderState_ = {};
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
	encoderState_.valid = false;
	pipelines_.Clear();
	samplers_.Clear();
}

void DrawEngineMetal::BeginFrame() {
	DrawEngineCommon::BeginFrame();
	gstate_c.Dirty(DIRTY_ALL_RENDER_STATE);
}

void DrawEngineMetal::Invalidate(InvalidationCallbackFlags flags) {
	// GE native state is always rebound. Thin3d may also have replaced texture bindings.
	encoderState_.valid = false;
	gstate_c.Dirty(DIRTY_ALL_RENDER_STATE | DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS);
}

bool DrawEngineMetal::ApplyDrawState(GEPrimitiveType prim, MetalDrawState *state, std::string *error) {
	pipelineState_ = {};
	if (!gstate.isModeClear()) {
		pipelineState_.Convert(draw_->GetShaderLanguageDesc().bitwiseOps, gstate_c.Use(GPU_USE_SHADER_BLENDING));
		if (pipelineState_.FramebufferRead()) {
			FBOTexState binding = FBO_TEX_NONE;
			ApplyFramebufferRead(&binding);
			auto &blend = pipelineState_.blendState;
			ApplyStencilReplaceAndLogicOpIgnoreBlend(blend.replaceAlphaWithStencil, blend);
			if (binding == FBO_TEX_COPY_BIND_TEX) {
				auto *vfb = framebufferManager_->GetCurrentRenderVFB();
				if (framebufferManager_->UseBufferedRendering() && vfb && vfb->fbo) {
					if (!framebufferManager_->BindFramebufferAsColorTexture(DRAW_BINDING_2ND_TEXTURE,
						vfb, BINDFBCOLOR_MAY_COPY | BINDFBCOLOR_UNCACHED, Draw::ALL_LAYERS)) {
						*error = "Metal shader blending could not bind its framebuffer copy";
						return false;
					}
				} else if (!framebufferManager_->UseBufferedRendering()) {
					if (!manager_->SnapshotBackbufferColor(DRAW_BINDING_2ND_TEXTURE, error)) {
						return false;
					}
				} else {
					*error = "Metal shader blending has no current PSP framebuffer";
					return false;
				}
				SamplerCacheKey key{};
				key.sClamp = true;
				key.tClamp = true;
				manager_->SetNativeSampler(DRAW_BINDING_2ND_TEXTURE, samplers_.GetOrCreate(manager_->Context().Device(), key, error));
				if (!error->empty()) {
					return false;
				}
			} else if (binding != FBO_TEX_READ_FRAMEBUFFER) {
				*error = "Metal shader blending has no framebuffer read path";
				return false;
			}
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
	// Skinning may have partially decoded vertices already. The software and
	// depth-raster paths also need the decoded vertices on the CPU.
	const bool directDecode = hardware && !dec_->skinInDecode && !useDepthRaster_ && ComputeNumVertsToDecode() > 0;
	if (!directDecode) {
		DecodeVerts(dec_, decoded_);
	}
	const bool hasColor = (lastVType_ & GE_VTYPE_COL_MASK) != GE_VTYPE_COL_NONE;
	if (gstate.isModeThrough()) {
		gstate_c.vertexFullAlpha &= hasColor || gstate.getMaterialAmbientA() == 255;
	} else {
		gstate_c.vertexFullAlpha &= ((hasColor && (gstate.materialupdate & 1)) || gstate.getMaterialAmbientA() == 255) &&
			(!gstate.isLightingEnabled() || gstate.getAmbientA() == 255);
	}
	TextureApplyResult texture;
	const bool textureEnabled = !gstate.isModeClear() && gstate.isTextureMapEnabled();
	const bool textureNeedsApply = textureEnabled && gstate_c.IsDirty(DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS);
	if (textureNeedsApply) {
		textureCache_->ResetGETextureBindings();
		gstate_c.Clean(DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS);
		gstate_c.dstSquared = false;
		texture = textureCache_->ApplyTexture(true);
	} else if (gstate.getTextureAddress(0) == (gstate.getFrameBufRawAddress() | 0x04000000)) {
		// A framebuffer clear may have changed texture memory without changing the texture registers.
		gstate_c.Dirty(DIRTY_TEXTURE_IMAGE);
	}
	Metal::UploadSlice decodedUpload;
	uint64_t decodedUploadGeneration = 0;
	size_t decodedIndexOffset = 0;
	int count, maxIndex;
	bool indexed;
	if (directDecode) {
		// The index generator needs vertex offsets, not the decoded bytes. Work
		// them out first so vertices and indices fit in a single upload slice.
		int decodedCount = 0;
		for (int i = 0; i < numDrawVerts_; ++i) {
			const DeferredVerts &draw = drawVerts_[i];
			drawVertexOffsets_[i] = decodedCount - draw.indexLowerBound;
			decodedCount += draw.indexUpperBound - draw.indexLowerBound + 1;
		}
		numDecodedVerts_ = decodedCount;
		DecodeIndsAndGetData(&prim, &count, &maxIndex, &indexed, false);
		numDecodedVerts_ = 0;
		auto &context = manager_->Context();
		const size_t vertexBytes = (size_t)decodedCount * dec_->GetDecVtxFmt().stride;
		decodedIndexOffset = (vertexBytes + sizeof(uint16_t) - 1) & ~(sizeof(uint16_t) - 1);
		const size_t uploadBytes = indexed ? decodedIndexOffset + (size_t)count * sizeof(uint16_t) : vertexBytes;
		decodedUpload = context.ReserveUpload(uploadBytes, error);
		if (!decodedUpload) {
			return false;
		}
		DecodeVerts(dec_, (uint8_t *)decodedUpload.buffer.contents + decodedUpload.offset);
		_dbg_assert_(numDecodedVerts_ == decodedCount);
		if (indexed) {
			memcpy((uint8_t *)decodedUpload.buffer.contents + decodedUpload.offset + decodedIndexOffset,
				decIndex_, (size_t)count * sizeof(uint16_t));
		}
		decodedUploadGeneration = context.CommandGeneration();
	} else {
		DecodeIndsAndGetData(&prim, &count, &maxIndex, &indexed, !hardware);
	}
	uint16_t *indices = decIndex_;
	const void *vertices = directDecode ? (const uint8_t *)decodedUpload.buffer.contents + decodedUpload.offset : decoded_;
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
	if (textureNeedsApply) {
		textureCache_->ApplySampler(texture, clipInfoFlags_ & ClipInfoFlags::FlatZ, transformed.pixelMapped);
	}
	if (action == SW_CULLED) {
		return true;
	}
	// Shared texture/depth conversions can leave an intermediate target bound.
	// Restore the GE target before choosing attachment formats or drawing.
	auto *vfb = framebufferManager_->GetCurrentRenderVFB();
	if (framebufferManager_->UseBufferedRendering() && vfb && vfb->fbo && manager_->RenderTarget() != vfb->fbo) {
		framebufferManager_->RebindFramebuffer("Metal GE target restore");
	} else if (!framebufferManager_->UseBufferedRendering() && !manager_->RestoreBackbufferTarget(error)) {
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
	ViewportAndScissor viewport;
	ConvertViewportAndScissor(framebufferManager_->GetDisplayLayoutConfigCopy(), framebufferManager_->UseBufferedRendering(),
		framebufferManager_->GetRenderWidth(), framebufferManager_->GetRenderHeight(),
		framebufferManager_->GetTargetBufferWidth(), framebufferManager_->GetTargetBufferHeight(), viewport);
	int x = std::max(0, viewport.scissorX);
	int y = std::max(0, viewport.scissorY);
	int w = std::min(framebufferManager_->GetRenderWidth(), viewport.scissorX + std::max(0, viewport.scissorW)) - x;
	int h = std::min(framebufferManager_->GetRenderHeight(), viewport.scissorY + std::max(0, viewport.scissorH)) - y;
	if (w <= 0 || h <= 0 || viewport.viewportW <= 0 || viewport.viewportH <= 0) {
		return true;
	}
	MetalDrawState state;
	if (!ApplyDrawState(prim, &state, error)) {
		return false;
	}
	const MetalGEVertexShader *vs;
	const MetalGEFragmentShader *fs;
	if (!shaderManager_->GetShaders(dec_->VertexType(), pipelineState_, hardware, clipInfoFlags_, &vs, &fs, error)) {
		return false;
	}
	const VShaderID vertexID = vs->id;
	const FShaderID fragmentID = fs->id;
	const DecVtxFormat *decodedFormat = hardware ? &dec_->GetDecVtxFmt() : nullptr;
	const MetalGEPipeline *pipeline = pipelines_.Request(*shaderManager_, vertexID, fragmentID, decodedFormat,
		state.blend, manager_->ColorFormat(), manager_->DepthStencilFormat(), error, manager_->SampleCount());
	if (!error->empty()) {
		return false;
	}
	const uint64_t shaderGeneration = shaderManager_->CacheGeneration();
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
	const uint32_t stride = hardware ? decodedFormat->stride : sizeof(TransformedVertex);
	const size_t vertexBytes = (size_t)vertexCount * stride;
	Metal::UploadSlice vb;
	Metal::UploadSlice ib;
	if (directDecode) {
		vb = decodedUpload;
		if (indexed) {
			ib = {vb.buffer, vb.offset + decodedIndexOffset};
		}
		if (decodedUploadGeneration != context.CommandGeneration()) {
			// An intervening readback can recycle the old ring slot. Decode again
			// from the queued PSP vertices instead of reading that old slot.
			const size_t uploadBytes = indexed ? decodedIndexOffset + (size_t)count * sizeof(uint16_t) : vertexBytes;
			vb = context.ReserveUpload(uploadBytes, error);
			if (vb) {
				auto *dest = (uint8_t *)vb.buffer.contents + vb.offset;
				int decodedCount = 0;
				for (int i = 0; i < numDrawVerts_; ++i) {
					const DeferredVerts &draw = drawVerts_[i];
					const int numVerts = draw.indexUpperBound - draw.indexLowerBound + 1;
					const auto *source = (const uint8_t *)draw.verts + draw.indexLowerBound * dec_->VertexSize();
					dec_->DecodeVerts(dest + decodedCount * stride, source, &draw.uvScale, numVerts);
					decodedCount += numVerts;
				}
				_dbg_assert_(decodedCount == vertexCount);
				if (indexed) {
					memcpy(dest + decodedIndexOffset, indices, (size_t)count * sizeof(uint16_t));
					ib = {vb.buffer, vb.offset + decodedIndexOffset};
				}
			}
		}
	} else if (indexed) {
		const size_t indexOffset = (vertexBytes + sizeof(uint16_t) - 1) & ~(sizeof(uint16_t) - 1);
		vb = context.ReserveUpload(indexOffset + (size_t)count * sizeof(uint16_t), error);
		if (vb) {
			memcpy((uint8_t *)vb.buffer.contents + vb.offset, vertices, vertexBytes);
			memcpy((uint8_t *)vb.buffer.contents + vb.offset + indexOffset, indices, (size_t)count * sizeof(uint16_t));
			ib = {vb.buffer, vb.offset + indexOffset};
		}
	} else {
		vb = context.Upload(vertices, vertexBytes, error);
	}
	if (!vb || (indexed && !ib)) {
		return false;
	}
	if (!pipeline || shaderGeneration != shaderManager_->CacheGeneration()) {
		pipeline = pipelines_.GetOrCreate(*shaderManager_, vertexID, fragmentID, decodedFormat,
			state.blend, manager_->ColorFormat(), manager_->DepthStencilFormat(), error, manager_->SampleCount());
	}
	if (!pipeline) {
		return false;
	}
	if (textureEnabled) {
		textureCache_->RestoreGETextureBindings();
	}
	if (!shaderManager_->BindUniforms(encoder, error) || !manager_->BindTextures(encoder, pipeline->textureMask, error)) {
		return false;
	}
	const uint64_t serial = manager_->RenderStateSerial();
	const bool sameEncoderState = encoderState_.valid && encoderState_.serial == serial;
	const NSUInteger stencil = transformed.setStencil ? transformed.stencilValue : state.stencilRef;
	const MTLViewport mtlViewport{viewport.viewportX, viewport.viewportY, viewport.viewportW, viewport.viewportH, 0, 1};
	const MTLScissorRect mtlScissor{(NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h};
	if (!sameEncoderState || encoderState_.pipeline != pipeline->state) {
		[encoder setRenderPipelineState:pipeline->state];
	}
	if (!sameEncoderState || encoderState_.depth != depth) {
		[encoder setDepthStencilState:depth];
	}
	if (!sameEncoderState || encoderState_.stencil != stencil) {
		[encoder setStencilReferenceValue:stencil];
	}
	if (!sameEncoderState || encoderState_.cull != state.cull) {
		[encoder setCullMode:state.cull];
	}
	if (!sameEncoderState) {
		[encoder setFrontFacingWinding:MTLWindingCounterClockwise];
	}
	if (!sameEncoderState || encoderState_.depthClip != state.depthClip) {
		[encoder setDepthClipMode:state.depthClip];
	}
	if (!sameEncoderState) {
		[encoder setDepthBias:0 slopeScale:0 clamp:0];
	}
	if (!sameEncoderState || !SameViewport(encoderState_.viewport, mtlViewport)) {
		[encoder setViewport:mtlViewport];
	}
	if (!sameEncoderState || !SameScissor(encoderState_.scissor, mtlScissor)) {
		[encoder setScissorRect:mtlScissor];
	}
	if (!sameEncoderState || encoderState_.blendColor != state.blendColor) {
		float blendColor[4];
		Uint8x4ToFloat4(blendColor, state.blendColor);
		[encoder setBlendColorRed:blendColor[0] green:blendColor[1] blue:blendColor[2] alpha:blendColor[3]];
	}
	encoderState_.valid = true;
	encoderState_.serial = serial;
	encoderState_.pipeline = pipeline->state;
	encoderState_.depth = depth;
	encoderState_.stencil = stencil;
	encoderState_.cull = state.cull;
	encoderState_.depthClip = state.depthClip;
	encoderState_.viewport = mtlViewport;
	encoderState_.scissor = mtlScissor;
	encoderState_.blendColor = state.blendColor;
	[encoder setVertexBuffer:vb.buffer offset:vb.offset atIndex:Metal::VERTEX_BUFFER_SLOT];
	MTLPrimitiveType topology = hardware && prim == GE_PRIM_TRIANGLE_STRIP ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle;
	const NSUInteger instances = gstate_c.Use(GPU_USE_SINGLE_PASS_STEREO) && manager_->RenderTarget()->Layers() > 1 ? 2 : 1;
	if (indexed) {
		[encoder drawIndexedPrimitives:topology indexCount:count indexType:MTLIndexTypeUInt16 indexBuffer:ib.buffer indexBufferOffset:ib.offset instanceCount:instances];
	} else {
		[encoder drawPrimitives:topology vertexStart:0 vertexCount:count instanceCount:instances];
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
		ResetAfterDrawInline();
		return;
	}
	ResetAfterDrawInline();
	if (framebufferManager_) {
		framebufferManager_->SetColorUpdated(gstate_c.skipDrawReason);
	}
	if (gpuCommon_) {
		gpuCommon_->NotifyFlush();
	}
}
