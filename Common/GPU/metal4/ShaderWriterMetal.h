// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "Common/GPU/ShaderWriter.h"
#include "Common/GPU/metal4/MetalBackend.h"
class MetalShaderWriter {
public:
	// Extensions are supported for both OpenGL ES and Vulkan (though of course, they're different).
	MetalShaderWriter(char *buffer, const MetalShaderLanguageDesc &lang, ShaderStage stage, Slice<const char *> extensions = Slice<const char *>(), ShaderWriterFlags flags = ShaderWriterFlags::NONE) : p_(buffer), lang_(lang), stage_(stage), flags_(flags) {
		buffer[0] = '\0';
		Preamble(extensions);
	}
	MetalShaderWriter(const MetalShaderWriter &) = delete;

	// I tried to call all three write functions "W", but only MSVC
	// managed to disentangle the ambiguities, so had to give up on that.

	// Assumes the input is zero-terminated.
	// C : Copies a buffer directly to the stream.
	template<size_t T>
	MetalShaderWriter &C(const char(&text)[T]) {
		memcpy(p_, text, T);
		p_ += T - 1;
		return *this;
	}
	// W: Writes a zero-terminated string to the stream.
	MetalShaderWriter &W(std::string_view text) {
		size_t len = text.size();
		memcpy(p_, text.data(), len);
		p_ += len;
		*p_ = '\0';
		return *this;
	}

	// F: Formats into the buffer.
	MetalShaderWriter &F(const char *format, ...);

	MetalShaderWriter &endl() {
		return C("\n");
	}

	// Useful for fragment shaders in GLES.
	// We always default integers to high precision.
	void HighPrecisionFloat();
	void LowPrecisionFloat();

	// NOTE: samplers must live for the rest of MetalShaderWriter's lifetime. No way to express that in C++ though :(
	void DeclareSamplers(Slice<SamplerDef> samplers);

	// Same as DeclareSamplers, but doesn't actually declare them.
	// This is currently only required by FragmentShaderGenerator.
	void ApplySamplerMetadata(Slice<SamplerDef> samplers);

	void ConstFloat(const char *name, float value);
	void SetFlags(ShaderWriterFlags flags) { flags_ |= flags; }
	ShaderWriterFlags Flags() const { return flags_; }
	void SetTexBindingBase(int base) { texBindingBase_ = base; }

	MetalShaderWriter &SampleTexture2D(const char *texName, const char *uv);
	MetalShaderWriter &SampleTexture2DOffset(const char *texName, const char *uv, int offX, int offY);
	MetalShaderWriter &LoadTexture2D(const char *texName, const char *integer_uv, int level);
	MetalShaderWriter &GetTextureSize(const char *szVariable, const char *texName);

	// Simple shaders with no special tricks.
	void BeginVSMain(Slice<InputDef> inputs, Slice<UniformDef> uniforms, Slice<VaryingDef> varyings);
	void BeginFSMain(Slice<UniformDef> uniforms, Slice<VaryingDef> varyings);

	// For simple shaders that output a single color, we can deal with this generically.
	void EndVSMain(Slice<VaryingDef> varyings);
	void EndFSMain(const char *vec4_color_variable);

	const MetalShaderLanguageDesc &Lang() const {
		return lang_;
	}

	void Rewind(size_t offset) {
		p_ -= offset;
	}

	// Can probably remove this
	char *GetPos() {
		return p_;
	}

private:
	// Several of the shader languages ignore samplers, beware of that.
	void DeclareSampler2D(const SamplerDef &def);
	void DeclareTexture2D(const SamplerDef &def);
	void DeclareUniforms(const Slice<UniformDef> &uniforms);
	const SamplerDef *GetSamplerDef(const char *name) const;

	void Preamble(Slice<const char *> extensions);

	char *p_;
	const MetalShaderLanguageDesc &lang_;
	const ShaderStage stage_;
	Slice<SamplerDef> samplers_;
	ShaderWriterFlags flags_ = ShaderWriterFlags::NONE;
	Slice<SamplerDef> samplerDefs_;
	int texBindingBase_ = 1;
};
