// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.
#pragma once
#include "Common/GPU/thin3d.h"
#include "Common/GPU/Shader.h"

struct MetalShaderLanguageDesc : ShaderLanguageDesc {
	explicit MetalShaderLanguageDesc(ShaderLanguage language = GLSL_VULKAN) : ShaderLanguageDesc(language) {}
	bool framebufferArrayTextures = true;
};
namespace Metal {
struct UVRect { float u0, v0, u1, v1; };
bool IsBackend(Draw::DrawContext *draw);
bool SupportsSpatialUpscaling(Draw::DrawContext *draw);
bool UpscaleBoundTexture(Draw::DrawContext *draw, int binding, int width, int height, UVRect *uv = nullptr);
inline bool IsBlockCompressed(Draw::DataFormat format, int *blockSize = nullptr) {
	if (format == Draw::DataFormat::ETC2_R8G8B8A1_UNORM_BLOCK) {
		if (blockSize) { *blockSize = 8; }
		return true;
	}
	return Draw::DataFormatIsBlockCompressed(format, blockSize);
}
}
