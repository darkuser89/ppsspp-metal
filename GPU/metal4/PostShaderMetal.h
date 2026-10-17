// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "Common/File/Path.h"
#include "GPU/Common/PostShader.h"
struct MetalTextureShaderInfo : TextureShaderInfo {
	bool metalSupported = false;
	std::vector<std::string> metalNoInlineFunctions;
};

void MetalReloadAllPostShaderInfo(Draw::DrawContext *draw);

const ShaderInfo *MetalGetPostShaderInfo(std::string_view name);
std::vector<const ShaderInfo *> MetalGetPostShaderChain(const std::string &name);
std::vector<const ShaderInfo *> MetalGetFullPostShadersChain(const std::vector<std::string> &names);
bool PostShaderChainRequires60FPS(const std::vector<const ShaderInfo *> &chain);
const std::vector<ShaderInfo> &MetalGetAllPostShaderInfo();

const MetalTextureShaderInfo *MetalGetTextureShaderInfo(std::string_view name);
const std::vector<MetalTextureShaderInfo> &MetalGetAllTextureShaderInfo();
void MetalRemoveUnknownPostShaders(std::vector<std::string> *names);

// Call this any time you alter the postshader list. It makes sure
// that "usePrevFrame" shaders are at the end, and that there's only one.
// It'll also enforce any similar future rules.
void MetalFixPostShaderOrder(std::vector<std::string> *names);

std::vector<TextureShaderInfo> MetalTextureShaderList();
bool MetalTextureShaderSupported(std::string_view name);
