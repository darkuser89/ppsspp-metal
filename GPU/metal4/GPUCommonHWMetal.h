// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once

#include "GPU/GPUCommon.h"
#include "GPU/metal4/FramebufferManagerMetal.h"
#include "GPU/metal4/TextureCachePolicyMetal.h"

class StringWriter;

// Shared GPUCommon implementation for the HW backends.
// Things that are irrelevant for SoftGPU should live here.
class GPUCommonHWMetal : public GPUCommon {
public:
	GPUCommonHWMetal(GraphicsContext *gfxCtx, Draw::DrawContext *draw);
	~GPUCommonHWMetal() override;

	// This can fail, and if so no render pass is active.
	void SetCurFramebufferDirty(bool dirty) override { curFramebufferDirty_ = dirty; }
	void BeginHostFrame(const DisplayLayoutConfig &config) override;
	void Reinitialize() override;
	bool PerformMemoryCopy(u32 dest, u32 src, int size, GPUCopyFlag flags) override;
	bool PerformMemorySet(u32 dest, u8 value, int size) override;
	bool PerformReadbackToMemory(u32 dest, int size) override;
	bool PerformWriteColorFromMemory(u32 dest, int size) override;
	void PerformWriteFormattedFromMemory(u32 addr, int size, int width, GEBufferFormat format, int rectWidth) override;
	bool PerformWriteStencilFromMemory(u32 dest, int size, WriteStencil flags) override;
	bool PerformReadbackStencilToMemory(u32 dest, int size) override;
	void PrepareCopyDisplayToOutput(const DisplayLayoutConfig &config) override;
	void CopyDisplayToOutput(const DisplayLayoutConfig &config) override;
	void DoState(PointerWrap &p) override;
	void DeviceLost() override;
	void DeviceRestore(Draw::DrawContext *draw) override;

	u32 CheckGPUFeatures() const override;

	// From GPUDebugInterface.
	bool GetCurrentFramebuffer(GPUDebugBuffer &buffer, GPUDebugFramebufferType type, int maxRes) override;
	bool GetCurrentDepthbuffer(GPUDebugBuffer &buffer) override;
	bool GetCurrentStencilbuffer(GPUDebugBuffer &buffer) override;
	bool GetOutputFramebuffer(GPUDebugBuffer &buffer) override;
	std::vector<const VirtualFramebuffer *> GetFramebufferList() const override;
	bool GetCurrentTexture(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) override;
	bool GetCurrentClut(GPUDebugBuffer &buffer) override;

	FramebufferManagerCommon *GetFramebufferManagerCommon() override {
		return framebufferManager_;
	}
	TextureCacheCommon *GetTextureCacheCommon() override {
		return textureCache_;
	}

	// Using string because it's generic - makes no assumptions on the size of the shader IDs of this backend.
	std::vector<std::string> DebugGetShaderIDs(DebugShaderType shader) override;
	std::string DebugGetShaderString(std::string id, DebugShaderType shader, DebugShaderStringType stringType) override;

	void SetDisplayFramebuffer(u32 framebuf, u32 stride, GEBufferFormat format) override;
	void InvalidateCache(u32 addr, int size, GPUInvalidationType type) override;

	u32 DrawSync(int mode) override;
	void FlushPendingDrawing() override;
	int ListSync(int listid, int mode) override;

	bool FramebufferDirty() override;
	bool FramebufferReallyDirty() override;

	void Execute_VertexTypeSkinning(u32 op, u32 diff);

	void Execute_Prim(u32 op, u32 diff);
	void Execute_Bezier(u32 op, u32 diff);
	void Execute_Spline(u32 op, u32 diff);
	void Execute_BlockTransferStart(u32 op, u32 diff);

	void Execute_TexSize0(u32 op, u32 diff);
	void Execute_TexLevel(u32 op, u32 diff);
	void Execute_LoadClut(u32 op, u32 diff);

	void Execute_TexFlush(u32 op, u32 diff);

	// TODO: Have these return an error code if they jump to a bad address. If bad, stop the FastRunLoop.
	typedef void (GPUCommonHWMetal::*CmdFunc)(u32 op, u32 diff);

	void FastRunLoop(DisplayList &list) override;
	void Execute_End(u32 op, u32 diff);
	void Execute_ImmVertexAlphaPrim(u32 op, u32 diff);
	void ExecuteOp(u32 op, u32 diff) override;

	bool PresentedThisFrame() const override;

private:
	void UpdateState(GPURunState state) {
		gpuState = state;
		if (state != GPUSTATE_RUNNING) {
			downcount = 0;
		}
	}

	void CheckDepthUsage(VirtualFramebuffer *vfb) override;
	void CheckFlushOp(int cmd, u32 diff);

protected:
	void FormatGPUStatsCommon(StringWriter &w);
	void UpdateCmdInfo() override;

	void PreExecuteOp(u32 op, u32 diff) override;
	void ClearCacheNextFrame() override;

	// Needs to be called on GPU thread, not reporting thread.
	void BuildReportingInfo() override;
	void UpdateMSAALevel(Draw::DrawContext *draw) override;

	void CheckDisplayResized() override;
	void CheckRenderResized(const DisplayLayoutConfig &config) override;
	void CheckConfigChanged(const DisplayLayoutConfig &config) override;

	u32 CheckGPUFeaturesLate(u32 features) const;

	FramebufferManagerMetal *framebufferManager_ = nullptr;
	MetalTextureCacheCommon *textureCache_ = nullptr;

	void FlushImm();
	void DoBlockTransfer(u32 skipDrawReason);
	void PopDLQueue();
	void DoExecuteCall(u32 target);
	void CompleteFailedList(DisplayList &list);

	int msaaLevel_ = 0;
	ShaderManagerCommon *shaderManager_ = nullptr;
	bool curFramebufferDirty_ = false;
};
