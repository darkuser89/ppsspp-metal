// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/FramebufferManagerCommon.h"
#include "GPU/metal3/TextureCachePolicyMetal.h"
class PresentationMetal;
class FramebufferManagerMetal : public FramebufferManagerCommon {
public:
	FramebufferManagerMetal(const FramebufferManagerMetal &) = delete;
	FramebufferManagerMetal &operator=(const FramebufferManagerMetal &) = delete;
	FramebufferManagerMetal(Draw::DrawContext *draw);
	virtual ~FramebufferManagerMetal();

	void SetTextureCache(MetalTextureCacheCommon *tc) {
		textureCache_ = tc;
		FramebufferManagerCommon::SetTextureCache(tc);
	}
	void SetShaderManager(ShaderManagerCommon * sm) {
		shaderManager_ = sm;
	}
	void SetDrawEngine(DrawEngineCommon *td) {
		drawEngine_ = td;
	}

	void Init(int msaaLevel);
	virtual void BeginFrame(const DisplayLayoutConfig &config);
	void SetDisplayFramebuffer(u32 framebuf, u32 stride, GEBufferFormat format);
	void DestroyFramebuf(VirtualFramebuffer *v);

	VirtualFramebuffer *DoSetRenderFrameBuffer(FramebufferHeuristicParams &params, u32 skipDrawReason);
	VirtualFramebuffer *SetRenderFrameBuffer(bool framebufChanged, int skipDrawReason, bool *changed) {
		// Inlining this part since it's so frequent.
		if (!framebufChanged && currentRenderVfb_) {
			currentRenderVfb_->last_frame_render = gpuStats.totals.numFlips;
			currentRenderVfb_->dirtyAfterDisplay = true;
			if (!skipDrawReason)
				currentRenderVfb_->reallyDirtyAfterDisplay = true;
			*changed = false;
			return currentRenderVfb_;
		} else {
			// This is so that we will be able to drive DoSetRenderFramebuffer with inputs
			// that come from elsewhere than gstate.
			FramebufferHeuristicParams inputs;
			GetFramebufferHeuristicInputs(&inputs, gstate);
			VirtualFramebuffer *vfb = DoSetRenderFrameBuffer(inputs, skipDrawReason);
			_dbg_assert_msg_(vfb, "DoSetRenderFramebuffer must return a valid framebuffer.");
			_dbg_assert_msg_(currentRenderVfb_, "DoSetRenderFramebuffer must set a valid framebuffer.");
			*changed = true;
			return vfb;
		}
	}
	void SetDepthFrameBuffer(bool isClearingDepth);

	void RebindFramebuffer(const char *tag);
	std::vector<const VirtualFramebuffer *> GetFramebufferList() const;

	void PrepareCopyDisplayToOutput(const DisplayLayoutConfig &config, bool reallyDirty);
	void CopyDisplayToOutput(const DisplayLayoutConfig &config);

	bool NotifyFramebufferCopy(u32 src, u32 dest, int size, GPUCopyFlag flags, u32 skipDrawReason);
	void PerformWriteFormattedFromMemory(u32 addr, int size, int width, GEBufferFormat fmt, int rectWidth);
	void WriteRectFromMemory(u32 addr, int size, int stride, GEBufferFormat fmt, int rectWidth);
	void UpdateFromMemory(u32 addr, int size);
	void ApplyClearToMemory(int x1, int y1, int x2, int y2, u32 clearColor);
	bool ReadbackStencilToMemory(u32 addr, int size);

	// We changed our depth mode, gotta start over.
	// Ideally, we should convert depth buffers here, not just clear them.
	void ClearAllDepthBuffers();

	// Returns true if it's sure this is a direct FBO->FBO transfer and it has already handle it.
	// In that case we hardly need to actually copy the bytes in VRAM, they will be wrong anyway (unless
	// read framebuffers is on, in which case this should always return false).
	// If this returns false, a memory copy will happen and NotifyBlockTransferAfter will be called.
	bool NotifyBlockTransferBefore(u32 dstBasePtr, int dstStride, int dstX, int dstY, u32 srcBasePtr, int srcStride, int srcX, int srcY, int w, int h, int bpp, u32 skipDrawReason);

	// This gets called after the memory copy, in case NotifyBlockTransferBefore returned false.
	// Otherwise it doesn't get called.
	void NotifyBlockTransferAfter(u32 dstBasePtr, int dstStride, int dstX, int dstY, u32 srcBasePtr, int srcStride, int srcX, int srcY, int w, int h, int bpp, u32 skipDrawReason);

	bool BindFramebufferAsColorTexture(int stage, VirtualFramebuffer *framebuffer, int flags, int layer);
	void ReadFramebufferToMemory(VirtualFramebuffer *vfb, int x, int y, int w, int h, RasterChannel channel, Draw::ReadbackMode mode);

	void DownloadFramebufferForClut(u32 fb_address, u32 loadBytes);
	bool DrawFramebufferToOutput(const DisplayLayoutConfig &config, const u8 *srcPixels, int srcStride, GEBufferFormat srcPixelFormat);

	// TODO: Should split into one that uses config, and one that doesn't.
	void DrawPixels(VirtualFramebuffer *vfb, int dstX, int dstY, const u8 *srcPixels, GEBufferFormat srcPixelFormat, int srcStride, int width, int height, RasterChannel channel, const char *tag);

	size_t NumVFBs() const { return vfbs_.size(); }

	u32 PrevDisplayFramebufAddr() const {
		return prevDisplayFramebuf_ ? prevDisplayFramebuf_->fb_address : 0;
	}
	u32 CurrentDisplayFramebufAddr() const {
		return displayFramebuf_ ? displayFramebuf_->fb_address : 0;
	}

	u32 DisplayFramebufAddr() const {
		return displayFramebufPtr_;
	}
	u32 DisplayFramebufStride() const {
		return displayStride_;
	}
	GEBufferFormat DisplayFramebufFormat() const {
		return displayFormat_;
	}

	bool UseBufferedRendering() const {
		return useBufferedRendering_;
	}

	void ForceUseBufferedRendering(bool buf) {
		useBufferedRendering_ = true;
	}

	// TODO: Maybe just include the last depth buffer address in this, too.
	bool MayIntersectFramebufferColor(u32 start) const {
		// Clear the cache/kernel bits.
		start &= 0x3FFFFFFF;
		if (Memory::IsVRAMAddress(start))
			start &= 0x041FFFFF;
		// Most games only have two framebuffers at the start.
		if (start >= framebufColorRangeEnd_ || start < PSP_GetVidMemBase()) {
			return false;
		}
		return true;
	}

	VirtualFramebuffer *GetCurrentRenderVFB() const {
		return currentRenderVfb_;
	}

	// This only checks for the color channel, and if there are multiple overlapping ones
	// with different color depth, this might get things wrong.
	// DEPRECATED FOR NEW USES - avoid whenever possible.
	VirtualFramebuffer *GetVFBAt(u32 addr) const;

	// This will only return exact matches of addr+stride+format.
	VirtualFramebuffer *GetExactVFB(u32 addr, int stride, GEBufferFormat format) const;

	// If this doesn't find the exact VFB, but one with a different color format with matching stride,
	// it'll resolve the newest one at address to the format requested, and return that.
	VirtualFramebuffer *ResolveVFB(u32 addr, int stride, GEBufferFormat format);

	// Utility to get the display VFB.
	VirtualFramebuffer *GetDisplayVFB();

	int GetRenderWidth() const { return currentRenderVfb_ ? currentRenderVfb_->renderWidth : 480; }
	int GetRenderHeight() const { return currentRenderVfb_ ? currentRenderVfb_->renderHeight : 272; }
	int GetTargetWidth() const { return currentRenderVfb_ ? currentRenderVfb_->width : 480; }
	int GetTargetHeight() const { return currentRenderVfb_ ? currentRenderVfb_->height : 272; }
	int GetTargetBufferWidth() const { return currentRenderVfb_ ? currentRenderVfb_->bufferWidth : 480; }
	int GetTargetBufferHeight() const { return currentRenderVfb_ ? currentRenderVfb_->bufferHeight : 272; }
	int GetTargetStride() const { return currentRenderVfb_ ? currentRenderVfb_->fb_stride : 512; }
	GEBufferFormat GetTargetFormat() const { return currentRenderVfb_ ? currentRenderVfb_->fb_format : displayFormat_; }

	void SetColorUpdated(int skipDrawReason) {
		if (currentRenderVfb_) {
			SetColorUpdated(currentRenderVfb_, skipDrawReason);
		}
	}
	void SetSafeSize(int w, int h);

	void NotifyRenderResized(const DisplayLayoutConfig &config, int msaaLevel);
	virtual void NotifyDisplayResized();
	void NotifyConfigChanged();

	void CheckPostShaders(const DisplayLayoutConfig &config);

	virtual void DestroyAllFBOs();

	virtual void DeviceLost();
	virtual void DeviceRestore(Draw::DrawContext *draw);

	Draw::Framebuffer *GetTempFBO(TempFBO reason, u16 w, u16 h);
	// A single-sampled copy of the first sample of each pixel of a multisampled framebuffer, or null if it isn't
	// multisampled or the backend can't. For reading colors as values, which averaging the samples would corrupt.
	Draw::Framebuffer *CopyColorSample0(VirtualFramebuffer *vfb);

	// Debug features
	virtual bool GetFramebuffer(u32 fb_address, int fb_stride, GEBufferFormat format, GPUDebugBuffer &buffer, int maxRes);
	// The 480x272 the display shows, wherever in a framebuffer it is.
	bool GetDisplayFramebuffer(GPUDebugBuffer &buffer, int maxRes);
	virtual bool GetDepthbuffer(u32 fb_address, int fb_stride, u32 z_address, int z_stride, GPUDebugBuffer &buffer);
	virtual bool GetStencilbuffer(u32 fb_address, int fb_stride, GPUDebugBuffer &buffer);
	virtual bool GetOutputFramebuffer(GPUDebugBuffer &buffer);

	const std::vector<VirtualFramebuffer *> &Framebuffers() const {
		return vfbs_;
	}

	Draw2D *GetDraw2D() {
		return &draw2D_;
	}

	// If a vfb with the target format exists, resolve it (run CopyToColorFromOverlappingFramebuffers).
	// If it doesn't exist, create it and do the same.
	// Returns the resolved framebuffer.
	VirtualFramebuffer *ResolveFramebufferColorToFormat(VirtualFramebuffer *vfb, GEBufferFormat newFormat);


	// If from==to, returns a copy pipeline.
	Draw2DPipeline *GetReinterpretPipeline(GEBufferFormat from, GEBufferFormat to, float *scaleFactorX);

	// Public to be used from the texture cache's depal shenanigans.
	void BlitUsingRaster(
		Draw::Framebuffer *src, float srcX1, float srcY1, float srcX2, float srcY2,
		Draw::Framebuffer *dest, float destX1, float destY1, float destX2, float destY2,
		bool linearFilter,
		int scaleFactor,  // usually unused, except for swizzle...
		Draw2DPipeline *pipeline, const char *tag);

	void ReleasePipelines();

	int GetMSAALevel() const {
		return msaaLevel_;
	}

	void DiscardFramebufferCopy() {
		currentFramebufferCopy_ = nullptr;
	}

	bool PresentedThisFrame() const;

	const std::vector<VirtualFramebuffer *> &GetVFBs() const {
		return vfbs_;
	}

	// Hack, only needed for non-buffered rendering.
	const DisplayLayoutConfig &GetDisplayLayoutConfigCopy() const {
		return displayLayoutConfigCopy_;
	}

	// For the debugger.
	inline int PeekBindSeqCount() const { return fbBindSeqCount_; }

protected:
	void ReadbackFramebuffer(VirtualFramebuffer *vfb, int x, int y, int w, int h, RasterChannel channel, Draw::ReadbackMode mode, u8 *shadow = nullptr) override;
	// Used for when a shader is required, such as GLES.
	virtual bool ReadbackStencilbuffer(Draw::Framebuffer *fbo, int x, int y, int w, int h, uint8_t *pixels, int pixelsStride, Draw::ReadbackMode mode);
	void SetViewport2D(int x, int y, int w, int h);
	Draw::Texture *MakePixelTexture(const u8 *srcPixels, GEBufferFormat srcPixelFormat, int srcStride, int width, int height);
	void DrawActiveTexture(float x, float y, float w, float h, float destW, float destH, float u0, float v0, float u1, float v1, int uvRotation, int flags);

	void CopyToColorFromOverlappingFramebuffers(VirtualFramebuffer *dest);
	void CopyToDepthFromOverlappingFramebuffers(VirtualFramebuffer *dest);

	bool UpdateRenderSize(int msaaLevel);

	void FlushBeforeCopy();
	virtual void DecimateFBOs();  // keeping it virtual to let D3D do a little extra

	// Used by ReadFramebufferToMemory and later framebuffer block copies
	void BlitFramebuffer(VirtualFramebuffer *dst, int dstX, int dstY, VirtualFramebuffer *src, int srcX, int srcY, int w, int h, int bpp, RasterChannel channel, const char *tag);

	void CopyFramebufferForColorTexture(VirtualFramebuffer *dst, VirtualFramebuffer *src, int flags, int layer, bool *partial);

	void EstimateDrawingSize(u32 fb_address, int fb_stride, GEBufferFormat fb_format, int viewport_width, int viewport_height, int region_width, int region_height, int scissor_width, int scissor_height, int &drawing_width, int &drawing_height);

	void NotifyRenderFramebufferCreated(VirtualFramebuffer *vfb);
	static void NotifyRenderFramebufferUpdated(VirtualFramebuffer *vfb);
	void NotifyRenderFramebufferSwitched(VirtualFramebuffer *prevVfb, VirtualFramebuffer *vfb, bool isClearingDepth);

	void BlitFramebufferDepth(VirtualFramebuffer *src, VirtualFramebuffer *dst, bool allowSizeMismatch = false);
	void CopyDepthRect(Draw::Framebuffer *src, int srcX, int srcY, Draw::Framebuffer *dst, int dstX, int dstY, int w, int h, int scaleFactor, const char *tag);
	void ShiftDepthForOffsetRendering(VirtualFramebuffer *vfb, int xOffset);
	void EnsureStencilWritePipeline(bool useExportShader);
	void DrawStencilWritePasses(GEBufferFormat format, int values, u8 usedBits, bool useExportShader);

	void ResizeFramebufFBO(VirtualFramebuffer *vfb, int w, int h, bool force = false, bool skipCopy = false);

	static bool ShouldDownloadFramebufferColor(const VirtualFramebuffer *vfb);
	static bool ShouldDownloadFramebufferDepth(const VirtualFramebuffer *vfb);
	void DownloadFramebufferOnSwitch(VirtualFramebuffer *vfb);
	VirtualFramebuffer *FindDisplayVFB(u32 fbaddr, u32 *offsetX, u32 *offsetY);
	bool ReadFramebufferForDebug(VirtualFramebuffer *vfb, int x, int y, int w, int h, GPUDebugBuffer &buffer, int maxScaleFactor);

	bool FindTransferFramebuffer(u32 basePtr, int stride, int x, int y, int w, int h, int bpp, bool destination, BlockTransferRect *rect);

	VirtualFramebuffer *FindDownloadTempBuffer(VirtualFramebuffer *vfb, RasterChannel channel);
	virtual void UpdateDownloadTempBuffer(VirtualFramebuffer *nvfb) {}

	VirtualFramebuffer *CreateRAMFramebuffer(uint32_t fbAddress, int width, int height, int stride, GEBufferFormat format);

	void UpdateFramebufUsage(VirtualFramebuffer *vfb) const;

	int GetFramebufferLayers() const;

	static void SetColorUpdated(VirtualFramebuffer *dstBuffer, int skipDrawReason) {
		dstBuffer->memoryUpdated = false;
		dstBuffer->readbackShadowValid = false;
		dstBuffer->clutUpdatedBytes = 0;
		dstBuffer->dirtyAfterDisplay = true;
		dstBuffer->drawnWidth = dstBuffer->width;
		dstBuffer->drawnHeight = dstBuffer->height;
		if ((skipDrawReason & SKIPDRAW_SKIPFRAME) == 0)
			dstBuffer->reallyDirtyAfterDisplay = true;
	}

	// For use when binding.
	inline int GetBindSeqCount() { return fbBindSeqCount_++; }

	static SkipGPUReadbackMode GetSkipGPUReadbackMode();

	PresentationMetal *presentation_ = nullptr;
	MetalTextureCacheCommon *textureCache_ = nullptr;
};
