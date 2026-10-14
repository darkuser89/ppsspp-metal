// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/ShaderCommon.h"
namespace ShaderBinding {
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
