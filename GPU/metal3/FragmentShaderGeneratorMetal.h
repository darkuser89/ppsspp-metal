// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/FragmentShaderGenerator.h"
#include "Common/GPU/metal3/MetalBackend.h"
bool GenerateMetalFragmentShader(const FShaderID &id, char *buffer, const MetalShaderLanguageDesc &compat, Draw::Bugs bugs, uint64_t *uniformMask, FragmentShaderFlags *fragmentShaderFlags, std::string *errorString, bool samplerLodBiasInShader = false);
