// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/ShaderManagerMetal.h"

#include "Core/Config.h"
#include "GPU/Common/GPUStateUtils.h"

// Mip bias is part of UB_VS_FS_Base but not DIRTY_BASE_UNIFORMS in the shared
// definitions. Keep it live in Metal even when it is the only changed uniform.
static constexpr uint64_t METAL_BASE_UNIFORMS = DIRTY_BASE_UNIFORMS | DIRTY_MIPBIAS;

ShaderManagerMetal::ShaderManagerMetal(Draw::DrawContext *draw) : ShaderManagerCommon(draw) {
	language_.framebufferArrayTextures = false;
	DeviceRestore(draw);
}

void ShaderManagerMetal::ClearShaders() {
	vertexCache_.clear();
	fragmentCache_.clear();
	++generation_;
	// A fresh generator environment can change the interpretation of uniforms.
	gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_ALL_UNIFORMS);
}

void ShaderManagerMetal::DeviceLost() {
	ClearShaders();
	buffers_ = {};
	base_ = {};
	lights_ = {};
	uniformOptionsValid_ = false;
	samplerLodBias_ = 0.0f;
	samplerLodBiasDirty_ = true;
	manager_ = nullptr;
	draw_ = nullptr;
}

void ShaderManagerMetal::DeviceRestore(Draw::DrawContext *draw) {
	DeviceLost();
	draw_ = draw;
	manager_ = draw ? (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER) : nullptr;
	useFlags_ = gstate_c.UseFlags();
	vendorChecks_ = g_Config.bVendorBugChecksEnabled;
}

bool ShaderManagerMetal::CheckEnvironment(std::string *error) {
	error->clear();
	if (!manager_ || !draw_) {
		*error = "Metal shader manager has no rendering device";
		return false;
	}
	if (useFlags_ != gstate_c.UseFlags() || vendorChecks_ != g_Config.bVendorBugChecksEnabled) {
		ClearShaders();
		useFlags_ = gstate_c.UseFlags();
		vendorChecks_ = g_Config.bVendorBugChecksEnabled;
	}
	return true;
}

bool ShaderManagerMetal::Compile(MetalGEShader *shader, ShaderStage stage, const char *tag) {
	Metal::ShaderCompileOptions options;
	options.textureBindingBase = 0;  // GE texture slots 0..2, uniforms 3..4.
#if PPSSPP_PLATFORM(IOS)
	options.ios = true;
#endif
	if (!Metal::CompileShader(shader->source, stage, options, &shader->compiled, &shader->error)) {
		return false;
	}
	for (const auto &resource : shader->compiled.resources) {
		if (resource.kind == Metal::ResourceKind::UniformBuffer) {
			size_t available = resource.binding == DRAW_BINDING_DYNUBO_BASE ? sizeof(base_) :
				resource.binding == DRAW_BINDING_DYNUBO_LIGHT ? sizeof(lights_) : 0;
			if (!available || resource.byteSize > available) {
				shader->error = "Generated GE uniform block does not match the Metal upload layout";
				return false;
			}
		} else if (resource.kind == Metal::ResourceKind::SampledTexture) {
			if (resource.index > DRAW_BINDING_DEPAL_TEXTURE) {
				shader->error = "Generated GE texture exceeds the Metal binding layout";
				return false;
			}
		} else {
			shader->error = "Generated GE shader uses an unsupported Metal resource kind";
			return false;
		}
	}
	shader->function = manager_->Context().CreateShader(shader->compiled, tag, &shader->error);
	return shader->function != nil;
}

const MetalGEVertexShader *ShaderManagerMetal::GetVertexShaderFromID(VShaderID id, std::string *error) {
	if (!CheckEnvironment(error)) {
		return nullptr;
	}
	if (id.is_invalid()) {
		*error = "Invalid Metal GE vertex shader ID";
		return nullptr;
	}
	auto found = vertexCache_.find(id);
	if (found == vertexCache_.end()) {
		auto shader = std::make_unique<MetalGEVertexShader>();
		shader->id = id;
		code_.fill(0);
		if (GenerateVertexShader(id, code_.data(), language_, draw_->GetBugs(), &shader->attributeMask,
			&shader->uniformMask, &shader->flags, &shader->error)) {
			shader->source = code_.data();
			Compile(shader.get(), ShaderStage::Vertex, id.Description().c_str());
		} else if (shader->error.empty()) {
			shader->error = "GE vertex shader generation failed";
		}
		found = vertexCache_.emplace(id, std::move(shader)).first;
	}
	if (!found->second->function) {
		*error = found->second->error;
		return nullptr;
	}
	return found->second.get();
}

const MetalGEFragmentShader *ShaderManagerMetal::GetFragmentShaderFromID(FShaderID id, std::string *error) {
	if (!CheckEnvironment(error)) {
		return nullptr;
	}
	if (id.is_invalid()) {
		*error = "Invalid Metal GE fragment shader ID";
		return nullptr;
	}
	auto found = fragmentCache_.find(id);
	if (found == fragmentCache_.end()) {
		auto shader = std::make_unique<MetalGEFragmentShader>();
		shader->id = id;
		code_.fill(0);
		if (GenerateFragmentShader(id, code_.data(), language_, draw_->GetBugs(), &shader->uniformMask,
			&shader->flags, &shader->error, true)) {
			shader->source = code_.data();
			Compile(shader.get(), ShaderStage::Fragment, id.Description().c_str());
		} else if (shader->error.empty()) {
			shader->error = "GE fragment shader generation failed";
		}
		found = fragmentCache_.emplace(id, std::move(shader)).first;
	}
	if (!found->second->function) {
		*error = found->second->error;
		return nullptr;
	}
	return found->second.get();
}

bool ShaderManagerMetal::GetShaders(u32 vertexType, const ComputedPipelineState &pipelineState, bool useHWTransform,
	ClipInfoFlags clipInfoFlags, const MetalGEVertexShader **vertex, const MetalGEFragmentShader **fragment, std::string *error) {
	*vertex = nullptr;
	*fragment = nullptr;
	if (!CheckEnvironment(error)) {
		return false;
	}
	VShaderID vsid;
	FShaderID fsid;
	// Include the draw's transform and clipping choices on every lookup. A clear
	// dirty bit alone does not guarantee that these per-draw arguments match.
	ComputeVertexShaderID(&vsid, vertexType, useHWTransform, clipInfoFlags);
	ComputeFragmentShaderID(&fsid, pipelineState, draw_->GetBugs(), clipInfoFlags);
	if (fsid.Bit(FS_BIT_FLATSHADE) != vsid.Bit(VS_BIT_FLATSHADE) ||
		fsid.Bit(FS_BIT_LMODE) != vsid.Bit(VS_BIT_LMODE) ||
		fsid.Bit(FS_BIT_MINMAX_DISCARD) != vsid.Bit(VS_BIT_FS_MINMAX_DISCARD) ||
		fsid.Bit(FS_BIT_DEPTH_CLAMP) != vsid.Bit(VS_BIT_FS_DEPTH_CLAMP)) {
		*error = "Metal GE vertex and fragment interfaces disagree";
		return false;
	}
	const auto vs = GetVertexShaderFromID(vsid, error);
	if (!vs) {
		return false;
	}
	const auto fs = GetFragmentShaderFromID(fsid, error);
	if (!fs) {
		return false;
	}
	*vertex = vs;
	*fragment = fs;
	gstate_c.Clean(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE);
	return true;
}

void ShaderManagerMetal::SetSamplerLodBias(float bias) {
	if (bias != samplerLodBias_) {
		samplerLodBias_ = bias;
		samplerLodBiasDirty_ = true;
	}
}

bool ShaderManagerMetal::UpdateUniforms(bool useBufferedRendering, bool pixelMapped, std::string *error) {
	if (!CheckEnvironment(error)) {
		return false;
	}
	uint64_t dirty = gstate_c.GetDirtyUniforms();
	if (!buffers_.base) {
		dirty |= METAL_BASE_UNIFORMS;
	}
	if (!buffers_.lights) {
		dirty |= DIRTY_LIGHT_UNIFORMS;
	}
	if (!uniformOptionsValid_ || buffered_ != useBufferedRendering || pixelMapped_ != pixelMapped) {
		dirty |= DIRTY_FRAMEBUFFER_DIM | DIRTY_PROJMATRIX | DIRTY_DEPAL;
	}
	UB_VS_FS_Base base = base_;
	UB_VS_Lights lights = lights_;
	MetalGEUniformBuffers next = buffers_;
	if ((dirty & METAL_BASE_UNIFORMS) || samplerLodBiasDirty_) {
		BaseUpdateUniforms(&base, dirty, useBufferedRendering, pixelMapped);
		base.samplerLodBias = samplerLodBias_;
		next.base = [manager_->Context().Device() newBufferWithBytes:&base length:sizeof(base) options:MTLResourceStorageModeShared];
		if (!next.base) {
			*error = "Failed to upload Metal GE base uniforms";
			return false;
		}
		next.base.label = @"GE base uniforms";
	}
	if (dirty & DIRTY_LIGHT_UNIFORMS) {
		LightUpdateUniforms(&lights, dirty);
		next.lights = [manager_->Context().Device() newBufferWithBytes:&lights length:sizeof(lights) options:MTLResourceStorageModeShared];
		if (!next.lights) {
			*error = "Failed to upload Metal GE light uniforms";
			return false;
		}
		next.lights.label = @"GE light uniforms";
	}
	base_ = base;
	lights_ = lights;
	buffers_ = next;
	buffered_ = useBufferedRendering;
	pixelMapped_ = pixelMapped;
	uniformOptionsValid_ = true;
	samplerLodBiasDirty_ = false;
	gstate_c.CleanUniforms();
	return true;
}

bool ShaderManagerMetal::BindUniforms(id<MTLRenderCommandEncoder> encoder, std::string *error) const {
	error->clear();
	if (!manager_ || !encoder || !buffers_.base || !buffers_.lights) {
		*error = "Metal GE uniform buffers or render encoder are unavailable";
		return false;
	}
	[encoder setVertexBuffer:buffers_.base offset:0 atIndex:DRAW_BINDING_DYNUBO_BASE];
	[encoder setFragmentBuffer:buffers_.base offset:0 atIndex:DRAW_BINDING_DYNUBO_BASE];
	[encoder setVertexBuffer:buffers_.lights offset:0 atIndex:DRAW_BINDING_DYNUBO_LIGHT];
	[encoder setFragmentBuffer:buffers_.lights offset:0 atIndex:DRAW_BINDING_DYNUBO_LIGHT];
	return true;
}

std::vector<std::string> ShaderManagerMetal::DebugGetShaderIDs(DebugShaderType type) {
	std::vector<uint64_t> ids;
	if (type == SHADER_TYPE_VERTEX) {
		for (const auto &entry : vertexCache_) {
			ids.push_back(entry.first.ToUint64());
		}
	} else if (type == SHADER_TYPE_FRAGMENT) {
		for (const auto &entry : fragmentCache_) {
			ids.push_back(entry.first.ToUint64());
		}
	}
	return ToSortedDebugShaderIdVec(ids);
}

std::string ShaderManagerMetal::DebugGetShaderString(std::string id, DebugShaderType type, DebugShaderStringType stringType) {
	// The common debugger IDs are eight raw bytes, not hexadecimal text.
	if (id.size() != sizeof(uint64_t)) {
		return "";
	}
	ShaderID parsed;
	parsed.FromString(id);
	const MetalGEShader *shader = nullptr;
	std::string description;
	if (type == SHADER_TYPE_VERTEX) {
		auto found = vertexCache_.find(VShaderID(parsed));
		if (found != vertexCache_.end()) {
			shader = found->second.get();
			description = found->first.Description();
		}
	} else if (type == SHADER_TYPE_FRAGMENT) {
		auto found = fragmentCache_.find(FShaderID(parsed));
		if (found != fragmentCache_.end()) {
			shader = found->second.get();
			description = found->first.Description();
		}
	}
	if (!shader) {
		return "";
	}
	switch (stringType) {
	case SHADER_STRING_SHORT_DESC: return description;
	case SHADER_STRING_SOURCE_CODE: return shader->compiled.source.empty() ? shader->source : shader->compiled.source;
	case SHADER_STRING_STATS: return shader->error.empty() ? "Native Metal GE shader" : shader->error;
	default: return "";
	}
}
