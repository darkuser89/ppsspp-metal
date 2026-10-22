// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.
#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include "Common/GPU/metal4/MetalShaderCompiler.h"

namespace Metal::TranslationCache {

constexpr uint64_t MAX_FILE_BYTES = 64 * 1024 * 1024;
constexpr uint32_t MAX_GLSL_BYTES = 32768;
constexpr uint32_t MAX_MSL_BYTES = 1024 * 1024;

inline uint64_t OptionsKey(ShaderStage stage, const ShaderCompileOptions &options) {
	// GE shaders do not request out-of-line helpers. Such callers bypass this cache.
	if (!options.noInlineFunctions.empty()) {
		return UINT64_MAX;
	}
	return (uint64_t)options.textureBindingBase << 16 | (uint64_t)stage << 8 |
		options.flipVertexY | (uint64_t)options.ios << 1 | (uint64_t)options.framebufferFetch << 2 |
		(uint64_t)options.multiview << 3;
}

struct Record {
	uint64_t options = 0;
	std::string glsl;
	CompiledShader compiled;
	bool Matches(std::string_view source, ShaderStage stage, const ShaderCompileOptions &compileOptions) const {
		return options != UINT64_MAX && options == OptionsKey(stage, compileOptions) && glsl == source;
	}
};

// Explicit fields, bounded allocations, and a payload budget excluding the checksum.
// The caller verifies the complete file checksum before constructing a Reader.
class Reader {
public:
	Reader(FILE *file, uint64_t remaining) : file_(file), remaining_(remaining) {}
	bool Bytes(void *data, size_t size) {
		if (size > remaining_ || fread(data, 1, size, file_) != size) {
			return false;
		}
		remaining_ -= size;
		return true;
	}
	template<class T> bool Value(T *value) { return Bytes(value, sizeof(*value)); }
	bool String(std::string *value, uint32_t maxSize) {
		uint32_t size;
		if (!Value(&size) || !size || size > maxSize || size > remaining_) {
			return false;
		}
		value->resize(size);
		return Bytes(value->data(), size) && value->find('\0') == std::string::npos;
	}
	uint64_t Remaining() const { return remaining_; }
private:
	FILE *file_;
	uint64_t remaining_;
};

inline bool Read(Reader &reader, Record *record) {
	uint32_t count, viewMask;
	if (!reader.Value(&record->options) || !reader.String(&record->glsl, MAX_GLSL_BYTES) ||
		!reader.String(&record->compiled.source, MAX_MSL_BYTES) || !reader.String(&record->compiled.entryPoint, 256) ||
		!reader.Value(&count) || count > 32) {
		return false;
	}
	record->compiled.resources.clear();
	std::array<uint32_t, 2> used{};
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t kind;
		ShaderResource resource{};
		if (!reader.Value(&kind) || !reader.Value(&resource.binding) || !reader.Value(&resource.index) ||
			!reader.Value(&resource.byteSize) || resource.binding != resource.index || resource.index >= 16 ||
			(kind != (uint32_t)ResourceKind::UniformBuffer && kind != (uint32_t)ResourceKind::SampledTexture)) {
			return false;
		}
		resource.kind = (ResourceKind)kind;
		const bool buffer = resource.kind == ResourceKind::UniformBuffer;
		if ((used[buffer] & (1u << resource.index)) || (buffer ? resource.byteSize > MAX_MSL_BYTES : resource.byteSize != 0)) {
			return false;
		}
		used[buffer] |= 1u << resource.index;
		record->compiled.resources.push_back(resource);
	}
	for (auto &size : record->compiled.workgroupSize) {
		if (!reader.Value(&size) || size != 1) {
			return false;
		}
	}
	if (!reader.Value(&viewMask) || viewMask > 1) {
		return false;
	}
	record->compiled.needsViewMaskBuffer = viewMask != 0;
	return true;
}

inline bool WriteString(FILE *file, const std::string &value, uint32_t maxSize) {
	const uint32_t size = (uint32_t)value.size();
	return !value.empty() && value.size() <= maxSize && fwrite(&size, sizeof(size), 1, file) == 1 &&
		fwrite(value.data(), 1, size, file) == size;
}

inline bool Write(FILE *file, uint64_t options, const std::string &glsl, const CompiledShader &compiled) {
	const uint32_t count = (uint32_t)compiled.resources.size();
	if (options == UINT64_MAX || compiled.resources.size() > 32 || fwrite(&options, sizeof(options), 1, file) != 1 ||
		!WriteString(file, glsl, MAX_GLSL_BYTES) || !WriteString(file, compiled.source, MAX_MSL_BYTES) ||
		!WriteString(file, compiled.entryPoint, 256) || fwrite(&count, sizeof(count), 1, file) != 1) {
		return false;
	}
	for (const auto &resource : compiled.resources) {
		const uint32_t fields[] = {(uint32_t)resource.kind, resource.binding, resource.index, resource.byteSize};
		if (fwrite(fields, sizeof(fields), 1, file) != 1) {
			return false;
		}
	}
	const uint32_t viewMask = compiled.needsViewMaskBuffer;
	return fwrite(compiled.workgroupSize, sizeof(compiled.workgroupSize), 1, file) == 1 &&
		fwrite(&viewMask, sizeof(viewMask), 1, file) == 1;
}

}  // namespace Metal::TranslationCache
