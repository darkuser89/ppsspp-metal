// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#import <QuartzCore/CAMetalLayer.h>

#include "Common/GPU/Metal/MetalRenderContext.h"

namespace Draw {
class Framebuffer;
}

namespace Metal {

// Native access shared by thin3d, the platform surface and the GE renderer.
// All calls belong to the rendering thread. Native draws must use RenderEncoder
// so they join the current target's pass. EndRenderPass must precede transfers;
// do not create another encoder directly while the render pass is open.
class RenderManager {
public:
	virtual ~RenderManager() = default;
	virtual RenderContext &Context() = 0;
	virtual id<MTLRenderCommandEncoder> RenderEncoder() = 0;
	virtual void EndRenderPass() = 0;
	virtual Draw::Framebuffer *RenderTarget() const = 0;
	virtual MTLPixelFormat ColorFormat() const = 0;
	virtual MTLPixelFormat DepthStencilFormat() const = 0;
	virtual int SampleCount() const = 0;
	virtual void SetNativeSampler(int slot, id<MTLSamplerState> sampler) = 0;
	// Uses the bindings tracked by thin3d, including framebuffer/depal passes.
	virtual bool BindTextures(id<MTLRenderCommandEncoder> encoder, uint32_t mask, std::string *error) = 0;

	// The platform owns layer layout and drawableSize; this retains the layer.
	// Passing nil detaches it and returns to offscreen rendering.
	virtual bool SetSurface(CAMetalLayer *layer, std::string *error) = 0;
	virtual void ResizeSurface() = 0;
};

}  // namespace Metal
