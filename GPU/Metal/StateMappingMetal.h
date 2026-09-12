// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/Metal/PipelineManagerMetal.h"
#include "GPU/Common/GPUStateUtils.h"

struct MetalDrawState {
	MetalGEBlendState blend;
	MetalGEDepthStencilState depthStencil;
	MTLCullMode cull = MTLCullModeNone;
	MTLDepthClipMode depthClip = MTLDepthClipModeClip;
	uint32_t blendColor = 0;
	uint8_t stencilRef = 0;
};

// pipelineState has already resolved shader blending and framebuffer reads.
bool ConvertMetalDrawState(GEPrimitiveType prim, const ComputedPipelineState &pipelineState,
	MetalDrawState *state, std::string *error);
