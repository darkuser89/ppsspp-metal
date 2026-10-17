// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/ShaderCommon.h"
struct alignas(16) MetalGERasterUniforms {
	uint32_t dither[4];
	float raster[3];  // Target format + 1 (zero disables conversion), render scale X/Y.
	uint32_t flags;
};
static_assert(sizeof(MetalGERasterUniforms) == 32);
enum : uint32_t {
	METAL_RASTER_DITHER = 1,
	METAL_RASTER_LINEAR_FILTER = 4,
};
namespace ShaderBinding {
constexpr uint32_t DRAW_BINDING_METAL_RASTER = 15;
enum {
	DRAW_BINDING_TEXTURE = 0,
	DRAW_BINDING_2ND_TEXTURE = 1,
	DRAW_BINDING_DEPAL_TEXTURE = 2,
	DRAW_BINDING_DYNUBO_BASE = 3,
	DRAW_BINDING_DYNUBO_LIGHT = 4,
	DRAW_BINDING_DYNUBO_BONE = 5,
	DRAW_BINDING_TESS_STORAGE_BUF = 6,
	DRAW_BINDING_TESS_STORAGE_BUF_WU = 7,
	DRAW_BINDING_TESS_STORAGE_BUF_WV = 8,
	DRAW_BINDING_COUNT = 9,
};
}  // namespace ShaderBinding
